import os
import random
from typing import List

import polars as pl
from tqdm import tqdm
from utils import get_cell_line_labels


def sort(df):
    # Sample DataFrame
    # Sorting by 'age' in ascending order and then by 'score' in descending order
    sorted_df = df.sort(by=["Chromosome", "Start", "End"])

    return sorted_df


def process_chromosome(
    chr: str,
    df: pl.DataFrame,
    cell_lines: List[str],
    window_size: int = 16_500,
) -> pl.DataFrame:
    new_df_rows = []
    n = len(df)

    # index, base_number, lock
    first = {"index": 0, "base": 0, "lock": False}
    last = {"index": 0, "base": 0, "lock": False}

    for i in tqdm(range(n - 1), desc=f"Processing {chr}"):
        row_i = df.row(i)
        start = row_i[2] - (window_size // 2)
        end = row_i[1] + (window_size // 2)

        # break if i is length of dataframe (i starts at 0 so this should be out of bounds)
        if i >= n - 1:
            break

        # if no lock, increase last index until base is less than end
        if not last["lock"]:
            while last["base"] < end:
                last["index"] += 1
                # Handle when chr changes or end of dataframe
                if last["index"] >= n - 1:
                    last["lock"] = True
                    break
                else:
                    last["base"] = df.row(last["index"])[1]

        # if no lock, increase first index until base is more than start
        if not first["lock"]:
            while first["base"] < start:
                first["index"] += 1
                first["base"] = df.row(first["index"])[2]

        overlapping_sources = []
        # take sources from first index to last index (no need for -1 because python is exclusive on the last index)
        for j in range(first["index"], last["index"]):
            overlapping_sources.append(df.row(j)[3])

        non_overlapping_sources = list(set(cell_lines) - set(overlapping_sources))

        # Create a new row if there are any non-overlapping sources.
        if non_overlapping_sources:
            chosen_source = random.choice(non_overlapping_sources)
            new_row = {
                "Chromosome": row_i[0],
                "Start": row_i[1],
                "End": row_i[2],
                "source": chosen_source,
                "labels": ",".join(non_overlapping_sources),
            }
            new_df_rows.append(new_row)

    return pl.DataFrame(new_df_rows)


def generate_negative_samples(
    positive: pl.DataFrame,
    cell_lines: List[str],
    window_size: int = 16_500,
) -> pl.DataFrame:
    chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

    queries = []

    for chr in chromosomes:
        lazy_chr_df = positive.lazy().filter(pl.col("Chromosome") == chr).lazy()
        processed_lazy_chr = lazy_chr_df.map_batches(
            lambda df, current_chr=chr: process_chromosome(
                current_chr, df, cell_lines, window_size
            )
        )
        queries.append(processed_lazy_chr)

    results = pl.collect_all(queries)

    # Combine results back into a single dataframe
    combined_df = pl.concat(results)

    return combined_df


if __name__ == "__main__":
    # Get list of cell_lines
    cell_lines = get_cell_line_labels(
        "/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines"
    )

    # Read positive.bed
    positive = pl.read_csv(
        "data/positive.bed",
        separator="\t",
        has_header=False,
        new_columns=["Chromosome", "Start", "End", "source", "labels"],
    )

    # Sort positive sample
    print("Sorting dataframe...")
    positive_sorted = sort(positive)

    # Generate negative samples
    negative = generate_negative_samples(
        positive=positive_sorted, cell_lines=cell_lines
    )

    # Write negative samples to file
    negative.write_csv("data/negative.bed", separator="\t", has_header=False)

    # Combine positive and new_df
    combined_df = pl.concat([positive, negative])

    # Write the combined dataframe to combined.bed
    combined_df.write_csv("data/combined.bed", separator="\t", has_header=False)
