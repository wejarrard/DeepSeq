import os
import random

import polars as pl
import pytest


def read_peak_file(filepath):
    return pl.read_csv(
        filepath,
        separator="\t",
        has_header=False,
        columns=[0, 1, 2],
        new_columns=["Chromosome", "Start", "End"],
    )


def has_overlap(df_row, peak_df):
    # print(dict(list(df_row.items())[:3]))
    # print(peak_df.filter(
    #         (peak_df["Chromosome"] == df_row["Chromosome"])
    #         & (peak_df["Start"] < df_row["End"])
    #         & (peak_df["End"] > df_row["Start"])
    #     )[0])
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
    cell_lines_directory = (
        "/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines"
    )
    sample_size = 10

    # The assertion will fail if there's no overlap for any sampled row
    assert check_overlap_for_sampled_rows(
        df, cell_lines_directory, sample_size
    ), "No overlap found for one or more sampled rows"
