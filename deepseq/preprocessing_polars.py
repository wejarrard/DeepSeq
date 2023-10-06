import os
import polars as pl


def check_overlap(row: pl.Series, df: pl.DataFrame, overlap: float) -> list:
    """Check for overlap in the DataFrame using a percentage."""
    start_tolerance = (row["end"] - row["start"]) * overlap
    end_tolerance = (row["end"] - row["start"]) * overlap

    return df[
        (df["chr_name"] == row["chr_name"])
        & (
            (
                df["start"].between(
                    row["start"] - start_tolerance, row["end"] + end_tolerance
                )
            )
            | (
                df["end"].between(
                    row["start"] - start_tolerance, row["end"] + end_tolerance
                )
            )
        )
    ].index.tolist()


def get_cell_line_labels(cell_lines_directory: str) -> list:
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
    ]


def consolidate_csvs(
    cell_lines_directory: str, overlap_fraction: float = 0.1
) -> pl.DataFrame:
    assert os.path.exists(
        cell_lines_directory
    ), f"{cell_lines_directory} does not exist."
    assert os.path.isdir(
        cell_lines_directory
    ), f"{cell_lines_directory} is not a directory."

    dfs = []

    cell_line_labels = get_cell_line_labels(cell_lines_directory)

    for folder in cell_line_labels:
        folder_path = os.path.join(cell_lines_directory, folder)

        # Loop through each CSV file in the folder
        for file in os.listdir(folder_path):
            if file.endswith((".bed", ".csv", "narrowPeak")):
                file_path = os.path.join(folder_path, file)
                df = pl.read_csv(
                    file_path,
                    separator="\t",
                    columns=[0, 1, 2],
                    new_columns=["chr_name", "start", "end"],
                )
                df["cell_line"] = folder
                df[
                    "labels"
                ] = folder  # Initialize labels column with the current cell_line

                print(f"Processing {file_path}...")
                for existing_df in dfs:
                    for index, row in df.iter_rows():
                        overlapping_indices = check_overlap(
                            row, existing_df, overlap_fraction
                        )
                        if overlapping_indices:
                            # Update overlapping rows in existing_df
                            existing_df.loc[
                                overlapping_indices, "labels"
                            ] = existing_df.loc[overlapping_indices, "labels"].apply(
                                lambda x: ",".join(
                                    set(x.split(",") + [row["cell_line"]])
                                )
                            )

                            # Collect labels from overlapping rows to update the current row in df
                            overlapping_labels = set(
                                existing_df.loc[overlapping_indices, "labels"]
                                .str.split(",")
                                .explode()
                                .tolist()
                            )

                            # Update the current row in df
                            df.at[index, "labels"] = ",".join(
                                set(row["labels"].split(",") + list(overlapping_labels))
                            )

                dfs.append(df)

    consolidated_df = pl.concat(dfs)
    return consolidated_df


def peaks_in_window(row, df, window):
    """Identify cell lines with a peak in the window around the given row."""
    start_tolerance = row["end"] - (window // 2)
    end_tolerance = row["start"] + (window // 2)

    overlapping = df[
        (df["chr_name"] == row["chr_name"])
        & (
            (df["start"].between(start_tolerance, end_tolerance))
            | (df["end"].between(start_tolerance, end_tolerance))
        )
    ]

    return overlapping["cell_line"].unique()


def generate_negative_rows(df, cell_lines_directory, window=16_500):
    all_cell_lines = get_cell_line_labels(cell_lines_directory)

    negatives = []

    for _, row in df.iterrows():
        overlapping_cell_lines = peaks_in_window(row, df, window)
        negative_cell_lines = set(all_cell_lines) - set(overlapping_cell_lines)

        if negative_cell_lines:
            neg_row = {
                "chr_name": row["chr_name"],
                "start": row["end"] - (window // 2),
                "end": row["start"] + (window // 2),
                "cell_line": "negative",
                "labels": ",".join(negative_cell_lines),
            }
            negatives.append(neg_row)

    negative_df = pd.DataFrame(negatives)
    return negative_df


if __name__ == "__main__":
    positive = consolidate_csvs("data/cell_lines", 0.1)
    negative = generate_negative_rows(positive, "data/cell_lines")
    df = pd.concat([positive, negative], ignore_index=True)
    df.to_csv("data/consolidated.bed", index=False, sep="\t", header=False)
