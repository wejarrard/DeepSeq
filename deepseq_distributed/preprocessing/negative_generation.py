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

    first_index = 0

    for i in tqdm(range(n), desc=f"Processing {chr}"):
        row_i = df.row(i, named=True)
        start = row_i["End"] - (window_size // 2)
        end = row_i["Start"] + (window_size // 2)

        # Update first_index to exclude rows where 'End' is less than 'start'
        while first_index < n and df.row(first_index, named=True)["End"] < start:
            first_index += 1

        # Filter the dataframe for overlapping rows within the window
        overlapping_sources = []
        for j in range(first_index, n):
            row_j = df.row(j, named=True)
            if row_j["Start"] >= end:
                break  # No need to check further as the dataframe is sorted
            if row_j["Chromosome"] == row_i["Chromosome"]:
                overlapping_sources.append(row_j["source"])

        # Determine non-overlapping sources
        non_overlapping_sources = list(set(cell_lines) - set(overlapping_sources))

        # Create a new row if there are any non-overlapping sources
        if non_overlapping_sources:
            chosen_source = random.choice(non_overlapping_sources)
            new_row = {
                "Chromosome": row_i["Chromosome"],
                "Start": row_i["Start"],
                "End": row_i["End"],
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

    print(positive_sorted)

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
