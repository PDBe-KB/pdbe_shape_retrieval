import argparse
import logging
import numpy as np
from numpy import linalg
from typing import Tuple
import trimesh
from pyFM import mesh


logger = logging.getLogger(__name__)


def read_vertices(file_mesh1, file_mesh2):
    mesh1 = trimesh.load_mesh(file_mesh1)
    mesh2 = trimesh.load_mesh(file_mesh2)
    # Extract vertex coordinates
    vertices_s1 = np.array(mesh1.vertices)
    vertices_s2 = np.array(mesh2.vertices)
    return vertices_s1, vertices_s2


def optimal_rotation_translation(
    A: np.ndarray, B: np.ndarray, allow_mirror: bool = False, weights: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """A, B - matrices 3*n, weights - vector n, result - (matrix 3*3, matrix 3*1)
    Find the optimal rotation matrix R and translation vector t for 3D superimposition of A onto B,
    where columns of A, B are coordinates of corresponding points.
    If allow_mirror == True, allow also improper rotation (i.e. mirroring + rotation).
    A_superimposed = R * A + t
    """
    if weights is not None:
        sumW = weights.sum()
        cA = (A * weights).sum(axis=1, keepdims=True) / sumW
        cB = (B * weights).sum(axis=1, keepdims=True) / sumW
    else:
        cA = np.mean(A, axis=1, keepdims=True)
        cB = np.mean(B, axis=1, keepdims=True)

    R = optimal_rotation(A - cA, B - cB, allow_mirror=allow_mirror, weights=weights)
    t = np.matmul(R, -cA) + cB
    return R, t


def optimal_rotation(
    A: np.ndarray, B: np.ndarray, allow_mirror: bool = False, weights: np.ndarray = None
) -> np.ndarray:
    """A, B - matrices 3*n, weights - vector n, result - matrix 3*3
    Find the optimal rotation matrix for 3D superimposition of A onto B,
    where columns of A, B are coordinates of corresponding points.
    If allow_mirror == True, allow also improper rotation (i.e. mirroring + rotation).
    """

    if weights is not None:
        A = A * weights.reshape((1, -1))
    H = A @ B.transpose()
    U, S, Vh = linalg.svd(H)
    R = (U @ Vh).transpose()
    if not allow_mirror and np.linalg.det(R) < 0:  # type: ignore  # mypy doesn't know .det
        Vh[-1, :] = -Vh[-1, :]
        R = (U @ Vh).transpose()
    return R


def calculate_rotation_translation(mesh1, mesh2, map_p2p):
    # read point to point correspondances file and save it into a list
    file_p2p21 = map_p2p
    with open(file_p2p21) as csvfile:
        p2p21 = csvfile.read().splitlines()
        p2p21 = np.asarray(p2p21, dtype=int)
    list_p2p = p2p21

    # Get lists of vertices for surface 1 and surface 2 using TriMesh from pyFM
    vertices_1, vertices_2 = read_vertices(mesh1, mesh2)

    if len(vertices_2) != len(list_p2p):
        logging.error(
            "something went wrong, the number of correspondaces should match the number of vertices in surface 2 "
        )

    matrix_A = []
    matrix_B = vertices_2

    for i in list_p2p:
        point_A = vertices_1[i]
        matrix_A.append(point_A)
    matrix_A = np.array(matrix_A)

    matrix_A = np.transpose(matrix_A)
    matrix_B = np.transpose(matrix_B)

    R, t = optimal_rotation_translation(matrix_A, matrix_B)

    return R, t


def calculate_rotation_translation_fixed(mesh1, mesh2, map_p2p):

    # Read p2p map (assume 1-based!)
    list_p2p = np.loadtxt(map_p2p, dtype=int) - 1

    vertices_1, vertices_2 = read_vertices(mesh1, mesh2)

    if len(vertices_2) != len(list_p2p):
        raise ValueError(
            "Number of correspondences must equal number of vertices in mesh2"
        )

    # Build correspondence matrices
    A = vertices_2  # TARGET points
    B = vertices_1[list_p2p]  # SOURCE points mapped from target

    # Transpose to 3xn
    A = A.T
    B = B.T

    # Compute transform to map TARGET -> SOURCE
    R, t = optimal_rotation_translation(A, B)

    return R, t


def compute_aligned_meshes(
    file_mesh1, file_mesh2, R, area_normalize=True, center=False, use_transpose=True
):
    """
    Load meshes using mesh.TriMesh, align mesh2 to mesh1 using rotation R,
    and return aligned TriMesh objects.
    """

    # -----------------------
    # Load TriMesh objects
    # -----------------------
    mesh1 = mesh.TriMesh(file_mesh1, area_normalize=area_normalize, center=center)
    mesh2 = mesh.TriMesh(file_mesh2, area_normalize=area_normalize, center=center)

    V1 = mesh1.vertlist.copy()
    V2 = mesh2.vertlist.copy()
    F1 = mesh1.facelist
    F2 = mesh2.facelist

    # -----------------------
    # Center explicitly (safe even if class already centers)
    # -----------------------
    c1 = V1.mean(axis=0)
    c2 = V2.mean(axis=0)

    V1c = V1 - c1
    V2c = V2 - c2

    # -----------------------
    # Apply rotation to mesh2
    # -----------------------
    if use_transpose:
        V2c = V2c @ R.T
    else:
        V2c = V2c @ R

    # -----------------------
    # Put mesh2 into mesh1 frame
    # -----------------------
    V1c = V1c + c1
    V2c = V2c + c1

    # -----------------------
    # Reconstruct TriMesh objects (IMPORTANT)
    # -----------------------
    mesh1_aligned = mesh.TriMesh.__new__(mesh.TriMesh)
    mesh2_aligned = mesh.TriMesh.__new__(mesh.TriMesh)

    mesh1_aligned.vertlist = V1c
    mesh1_aligned.facelist = F1

    mesh2_aligned.vertlist = V2c
    mesh2_aligned.facelist = F2

    return mesh1_aligned, mesh2_aligned


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_p2p21",
        help="Input file with point to point correspondace",
        required=True,
    )
    parser.add_argument(
        "--input_mesh1",
        help="Input triangulated mesh 1",
        required=True,
    )
    parser.add_argument(
        "--input_mesh2",
        help="Input triangulated mesh 2",
        required=True,
    )

    args = parser.parse_args()

    # read point to point correspondances file and save it into a list
    file_p2p21 = args.input_p2p21
    with open(file_p2p21) as csvfile:
        p2p21 = csvfile.read().splitlines()
        p2p21 = np.asarray(p2p21, dtype=int)
    list_p2p = p2p21

    # Get lists of vertices for surface 1 and surface 2 using TriMesh from pyFM
    vertices_1, vertices_2 = read_vertices(args.input_mesh1, args.input_mesh2)

    if len(vertices_2) != len(list_p2p):
        logging.error(
            "something went wrong, the number of correspondaces should match the number of vertices in surface 2 "
        )

    matrix_A = []
    matrix_B = vertices_2

    for i in list_p2p:
        point_A = vertices_1[i]
        matrix_A.append(point_A)
    matrix_A = np.array(matrix_A)

    matrix_A = np.transpose(matrix_A)
    matrix_B = np.transpose(matrix_B)

    R, t = optimal_rotation_translation(matrix_A, matrix_B)
    logger.info("Translation vector: %s", t)
    logger.info("Rotation matrix: %s", R)


if __name__ == "__main__":
    main()
