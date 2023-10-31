import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Union

import pandas as pd
import pyranges as pr
from tqdm import tqdm
from utils import get_cell_line_labels

warnings.filterwarnings("ignore", category=FutureWarning)


def process_row(gr):
    # Create a PyRanges object for the current row
    current_range = pr.PyRanges(gr.to_frame().T)

    # Find overlaps with the rest of the intervals
    overlaps = all_gr.overlap(
        current_range
    )  # Note: all_gr needs to be globally accessible

    # Extract the names of the overlapping segments
    overlapping_names = overlaps.df["source"].unique().tolist()

    # Create a record for the current row
    consolidated_entry = {
        "Chromosome": gr["Chromosome"],
        "Start": gr["Start"],
        "End": gr["End"],
        "source": gr["source"],
        "labels": ", ".join(overlapping_names),
    }

    return consolidated_entry


def consolidate_csvs(
    cell_lines_directory: str, num_cores=1, overlap: Union[float, str] = "any"
) -> pd.DataFrame:
    assert os.path.exists(
        cell_lines_directory
    ), f"{cell_lines_directory} does not exist."
    assert os.path.isdir(
        cell_lines_directory
    ), f"{cell_lines_directory} is not a directory."

    cell_line_labels = get_cell_line_labels(cell_lines_directory)

    chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

    all_pyranges = []
    for folder in cell_line_labels:
        try:
            folder_path = os.path.join(cell_lines_directory, folder, "peaks")

            for file in os.listdir(folder_path):
                if file.endswith("filtered.broadPeak"):
                    file_path = os.path.join(folder_path, file)

                    # Adjust the column names to meet pyranges' requirements
                    df = pd.read_csv(
                        file_path,
                        delimiter="\t",
                        usecols=[0, 1, 2],
                        header=None,
                        names=["Chromosome", "Start", "End"],
                    )

                    df = df[df["Chromosome"].isin(chromosomes)]

                    df[
                        "source"
                    ] = folder  # This assumes 'folder' is your label; change as needed.

                    # Create a PyRanges object from the DataFrame
                    gr = pr.PyRanges(df)

                    all_pyranges.append(gr)
        except:
            print(f"Exception for {folder}")
            continue

    # Concatenate all PyRanges objects
    global all_gr
    all_gr = pr.concat(all_pyranges)

    # We will use as many workers as there are processor cores
    print("Starting BED processing...")
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # The map method applies the function to every item in the iterable, in this case, the DataFrame rows.
        # This is done in parallel, and the results are yielded as soon as they are finished. Might be better if
        # we use batching, but current speed is acceptable (~5 hours at 32 cores)
        results = list(
            tqdm(
                executor.map(process_row, [row for _, row in all_gr.df.iterrows()]),
                total=all_gr.df.shape[0],
            )
        )

    # Each result is a record, so we can just create our DataFrame from these results.
    consolidated_df = pd.DataFrame.from_records(results)

    return consolidated_df


if __name__ == "__main__":
    num_cores = 32

    positive = consolidate_csvs(
        "/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines",
        num_cores=num_cores,
    )
    positive.to_csv("positive.bed", index=False, sep="\t", header=False)
