import os
import random
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import List, Union

import pandas as pd
import pyranges as pr
from tqdm import tqdm

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
        "labels": ", ".join(overlapping_names),  # Change this formatting as needed
    }

    return consolidated_entry


def get_cell_line_labels(cell_lines_directory: str) -> List[str]:
    assert os.path.exists(
        cell_lines_directory
    ), f"{cell_lines_directory} does not exist."
    assert os.path.isdir(
        cell_lines_directory
    ), f"{cell_lines_directory} is not a directory."

    return [
        folder
        for folder in os.listdir(cell_lines_directory)
        if os.path.isdir(os.path.join(cell_lines_directory, folder))
        and folder != "metadata"
    ]


def consolidate_csvs(
    cell_lines_directory: str, num_cores=1, overlap: Union[float, str] = "any"
) -> pd.DataFrame:
    assert os.path.exists(
        cell_lines_directory
    ), f"{cell_lines_directory} does not exist."
    assert os.path.isdir(
        cell_lines_directory
    ), f"{cell_lines_directory} is not a directory."

    all_pyranges = []  # List to store PyRanges objects

    cell_line_labels = get_cell_line_labels(
        cell_lines_directory
    )  # Retrieve your cell line labels

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
    global all_gr  # Make this a global variable so it's accessible in process_row
    all_gr = pr.concat(all_pyranges)

    # We will use as many workers as there are processor cores
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # The map method applies the function to every item in the iterable, in this case, the DataFrame rows.
        # This is done in parallel, and the results are yielded as soon as they are finished.
        results = list(
            tqdm(
                executor.map(process_row, [row for _, row in all_gr.df.iterrows()]),
                total=all_gr.df.shape[0],
            )
        )

    # Each result is a record, so we can just create our DataFrame from these results.
    consolidated_df = pd.DataFrame.from_records(results)

    return consolidated_df


# Negative sample generation


def peaks_in_window(args: tuple) -> tuple:
    row, df, window = args
    start_tolerance = row["end"] - (window // 2)
    end_tolerance = row["start"] + (window // 2)

    overlapping = df[
        (df["chr_name"] == row["chr_name"])
        & (
            (df["start"].between(start_tolerance, end_tolerance))
            | (df["end"].between(start_tolerance, end_tolerance))
        )
    ]

    # Return the unique cell lines along with the row to preserve the connection
    return row, overlapping["cell_line"].unique()


def generate_negative_rows(
    df, cell_lines_directory, window=16_500, num_cores=1
) -> pd.DataFrame:
    all_cell_lines = get_cell_line_labels(cell_lines_directory)

    # Prepare the arguments for each function call in a list
    tasks = [(row, df, window) for _, row in df.iterrows()]

    negatives = []

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Using tqdm for a progress bar
        results = list(tqdm(executor.map(peaks_in_window, tasks), total=len(tasks)))

    for row, overlapping_cell_lines in results:
        negative_cell_lines = list(set(all_cell_lines) - set(overlapping_cell_lines))

        if negative_cell_lines:
            random_cell_line = random.choice(negative_cell_lines)

            neg_row = {
                "chr_name": row["chr_name"],
                "start": row["end"] - (window // 2),
                "end": row["start"] + (window // 2),
                "cell_line": random_cell_line,
                "labels": ",".join(negative_cell_lines),
            }
            negatives.append(neg_row)

    negative_df = pd.DataFrame(negatives)
    return negative_df


if __name__ == "__main__":
    num_cores = 24

    positive = consolidate_csvs(
        "/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines",
        num_cores=num_cores,
    )
    positive.to_csv("positive.bed", index=False, sep="\t", header=False)

    negative = generate_negative_rows(
        df=positive,
        cell_lines_directory="/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines",
        num_cores=num_cores,
    )
    negative.to_csv("negative.bed", index=False, sep="\t", header=False)

    df = pd.concat([positive, negative], ignore_index=True)
    df.to_csv("combined.bed", index=False, sep="\t", header=False)
