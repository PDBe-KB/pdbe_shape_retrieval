from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import numpy as np
from scipy.spatial import KDTree


logger = logging.getLogger(__name__)


def save_data_to_csv(data: np.ndarray, output_file: str) -> None:
    """Dump matrix data into a output csv file.
    Args:
        data (dict): The data (for spectral descriptors or functional maps) to be dumped.
        output_file (str): The path to the output file.

    Raises:
        ValueError: If `data` is empty.
    """
    if len(data) != 0:
        try:
            descr_data = tuple(map(tuple, data))
            df = pd.DataFrame(descr_data)
            df.to_csv(output_file, index=False, header=None)

            # return df

        except Exception as e:
            logger.error(
                "Invalid data frame for wks descriptors: probably wrong fields in the data"
            )
            logger.error(e)
    else:
        logger.info("No data found to save")

        return None


def save_list_to_csv(data: Any, output_file: str) -> None:
    """Dump list into a csv output file.
    Args:
        data (dict): The data for list of parameters to be dumped.
        output_file (str): The path to the output file.

    Raises:
        ValueError: If `data` is empty.
    """
    if len(data) != 0:
        try:
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False, header=None)

        except Exception as e:
            logger.error(
                "Invalid data frame list of parameters: probably wrong fields in the data"
            )
            logger.error(e)
    else:
        logger.info("No data found to save")

        return None


def find_minimum_distance_meshes(mesh1: Any, mesh2: Any) -> float:
    """
    Compute the minimum Euclidean distance between two meshes.

    Args:
        mesh1: A mesh object with a `vertlist` attribute containing vertex coordinates
               as an (N, 3) array.
        mesh2: A mesh object with a `vertlist` attribute containing vertex coordinates
               as an (M, 3) array.

    Returns:
        float: The minimum Euclidean distance between any vertex of `mesh1` and any
               vertex of `mesh2`.
    """
    # Get the vertices of each mesh
    vertices1 = mesh1.vertlist
    vertices2 = mesh2.vertlist

    # Use a KDTree for efficient nearest neighbor search
    tree = KDTree(vertices2)
    distances, _ = tree.query(vertices1)

    # Return the minimum distance
    return float(np.min(distances))
