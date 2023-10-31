import os
from typing import List


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
