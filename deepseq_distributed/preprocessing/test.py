import os

import polars as pl

### POSITIVE TEST


def read_peak_file(filepath):
    return pl.read_csv(
        filepath,
        separator="\t",
        has_header=False,
        columns=[0, 1, 2],
        new_columns=["Chromosome", "Start", "End"],
    )


def has_overlap(df_row, peak_df):
    return (
        peak_df.filter(
            (peak_df["Chromosome"] == df_row["Chromosome"])
            & (peak_df["Start"] < df_row["End"])
            & (peak_df["End"] > df_row["Start"])
        ).shape[0]
        > 0
    )


def check_overlap_for_sampled_rows(df, cell_lines_directory, sample_size):
    sampled_df = df.sample(n=sample_size)
    results = []

    for row in sampled_df.iter_rows(named=True):
        labels = row["labels"].split(", ")
        for label in labels:
            folder_path = os.path.join(cell_lines_directory, label, "peaks")
            for filename in os.listdir(folder_path):
                if filename.endswith("filtered.broadPeak"):
                    peak_df = read_peak_file(os.path.join(folder_path, filename))
                    results.append(has_overlap(row, peak_df))

    return all(results)


def test_positive_data_validation():
    df = pl.read_csv(
        "data/positive.bed",
        separator="\t",
        has_header=False,
        new_columns=["Chromosome", "Start", "End", "source", "labels"],
    )
    print(df)
    cell_lines_directory = (
        "/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines"
    )
    sample_size = 10

    # The assertion will fail if there's no overlap for any sampled row
    assert check_overlap_for_sampled_rows(
        df, cell_lines_directory, sample_size
    ), "One or more cell lines do not have an overlap with our consolidated file for one or more sampled rows"

    assert not check_no_overlap_for_sampled_rows(
        df, cell_lines_directory, sample_size
    ), "One or more cell lines have an overlap with our consolidated file for one or more sampled rows"


### NEGATIVE TEST


def check_no_overlap_for_sampled_rows(df, cell_lines_directory, sample_size):
    sampled_df = df.sample(n=sample_size)
    results = []

    for row in sampled_df.iter_rows(named=True):
        labels = row["labels"].split(",")
        for label in labels:
            folder_path = os.path.join(cell_lines_directory, label, "peaks")
            for filename in os.listdir(folder_path):
                if filename.endswith("filtered.broadPeak"):
                    file_path = os.path.join(folder_path, filename)
                    # Check if file is empty by attempting to read the first byte
                    if os.path.getsize(file_path) == 0:
                        # If the file is empty, skip this label
                        continue
                    peak_df = read_peak_file(file_path)
                    # Check if peak_df is empty
                    if peak_df.height == 0:
                        # If the DataFrame is empty, skip this label
                        continue
                    # Invert the result of has_overlap for the negative check
                    results.append(not has_overlap(row, peak_df))

    return all(results)


def test_negative_data_validation():
    df = pl.read_csv(
        "data/negative.bed",
        separator="\t",
        has_header=False,
        new_columns=["Chromosome", "Start", "End", "source", "labels"],
    )
    cell_lines_directory = (
        "/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines"
    )
    sample_size = 10

    # The assertion will fail if there's an overlap for any sampled row
    assert check_no_overlap_for_sampled_rows(
        df, cell_lines_directory, sample_size
    ), "One or more cell lines have an overlap with our consolidated file for one or more sampled rows"

    assert not check_overlap_for_sampled_rows(
        df, cell_lines_directory, sample_size
    ), "One or more cell lines do not have an overlap with our consolidated file for one or more sampled rows"


# compare dataframes to make sure no overlap

# def check_no_overlap_between_top_sorted_rows(positive_df, negative_df):
#     # Sort both dataframes by Chromosome, Start, and End
#     positive_sorted = positive_df.sort(["Chromosome", "Start", "End"])
#     negative_sorted = negative_df.sort(["Chromosome", "Start", "End"])

#     # Get the top 10 rows from both sorted dataframes
#     top_positive = positive_sorted.head(10)
#     top_negative = negative_sorted.head(10)

#     # Check for overlaps
#     for pos_row in top_positive.iter_rows(named=True):
#         for neg_row in top_negative.iter_rows(named=True):
#             # Check if there's any overlap
#             if (pos_row["Chromosome"] == neg_row["Chromosome"] and
#                 pos_row["Start"] < neg_row["End"] and
#                 pos_row["End"] > neg_row["Start"]):
#                 # If there's an overlap, return False
#                 return False

#     # If no overlaps are found, return True
#     return True

# def test_no_overlap_between_positive_and_negative():
#     # Read in the positive dataset
#     positive_df = pl.read_csv(
#         "data/positive.bed",
#         separator="\t",
#         has_header=False,
#         new_columns=["Chromosome", "Start", "End", "source", "labels"],
#     )

#     # Read in the negative dataset
#     negative_df = pl.read_csv(
#         "data/negative.bed",
#         separator="\t",
#         has_header=False,
#         new_columns=["Chromosome", "Start", "End", "source", "labels"],
#     )

#     # Ensure there is no overlap between the top sorted rows of the positive and negative datasets
#     assert check_no_overlap_between_top_sorted_rows(positive_df, negative_df), \
#         "Overlap found between the top sorted rows of the positive and negative datasets"
