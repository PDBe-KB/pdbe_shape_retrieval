import argparse
import logging
import os
import csv
import numpy as np
from numpy import linalg
from typing import Tuple
import trimesh
import torch

def read_vertices(file_mesh1,file_mesh2):
    mesh1 = trimesh.load_mesh(file_mesh1)
    mesh2 = trimesh.load_mesh(file_mesh2)
    # Extract vertex coordinates
    vertices_s1 = np.array(mesh1.vertices)
    vertices_s2 = np.array(mesh2.vertices)
    return vertices_s1, vertices_s2
def optimal_rotation_translation(A: np.ndarray, B: np.ndarray, allow_mirror: bool=False, weights: np.ndarray=None) -> Tuple[np.ndarray, np.ndarray]:
    '''  A, B - matrices 3*n, weights - vector n, result - (matrix 3*3, matrix 3*1)
    Find the optimal rotation matrix R and translation vector t for 3D superimposition of A onto B,
    where columns of A, B are coordinates of corresponding points.
    If allow_mirror == True, allow also improper rotation (i.e. mirroring + rotation).
    A_superimposed = R * A + t
    '''
    if weights is not None:
        sumW = weights.sum()
        cA = (A * weights).sum(axis=1, keepdims=True) / sumW
        cB = (B * weights).sum(axis=1, keepdims=True) / sumW
    else:
        cA = np.mean(A, axis=1, keepdims=True)
        cB = np.mean(B, axis=1, keepdims=True)
        
    R = optimal_rotation(A - cA, B - cB, allow_mirror=allow_mirror, weights=weights)
    t = np.matmul(R, -cA) + cB
    return R,t
def optimal_rotation(A: np.ndarray, B: np.ndarray, allow_mirror: bool=False, weights: np.ndarray=None) -> np.ndarray:
    ''' A, B - matrices 3*n, weights - vector n, result - matrix 3*3
    Find the optimal rotation matrix for 3D superimposition of A onto B,
    where columns of A, B are coordinates of corresponding points.
    If allow_mirror == True, allow also improper rotation (i.e. mirroring + rotation).
    '''

    
    if weights is not None:
        A = A * weights.reshape((1, -1))
    H = A @ B.transpose()
    U, S, Vh = linalg.svd(H)
    R = (U @ Vh).transpose()
    if not allow_mirror and np.linalg.det(R) < 0:  # type: ignore  # mypy doesn't know .det
        Vh[-1,:] = -Vh[-1,:]
        R = (U @ Vh).transpose()
    return R

def calculate_rotation_translation(mesh1,mesh2,map_p2p):
    #read point to point correspondances file and save it into a list
    file_p2p21 = map_p2p
    with open(file_p2p21) as csvfile:
           p2p21 = csvfile.read().splitlines()
           p2p21 = np.asarray(p2p21, dtype=int)
    list_p2p=p2p21

    #Get lists of vertices for surface 1 and surface 2 using TriMesh from pyFM 
    vertices_1, vertices_2 = read_vertices(mesh1,mesh2)
    
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

    R,t = optimal_rotation_translation(matrix_A,matrix_B)
    
    return R,t

def calculate_rotation_translation_fixed(mesh1, mesh2, map_p2p):

    # Read p2p map (assume 1-based!)
    list_p2p = np.loadtxt(map_p2p, dtype=int) - 1

    vertices_1, vertices_2 = read_vertices(mesh1, mesh2)

    if len(vertices_2) != len(list_p2p):
        raise ValueError("Number of correspondences must equal number of vertices in mesh2")

    # Build correspondence matrices
    A = vertices_2  # TARGET points
    B = vertices_1[list_p2p]  # SOURCE points mapped from target

    # Transpose to 3xn
    A = A.T
    B = B.T

    # Compute transform to map TARGET -> SOURCE
    R, t = optimal_rotation_translation(A, B)

    return R, t
def main():
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
    
    #read point to point correspondances file and save it into a list
    file_p2p21 = args.input_p2p21
    with open(file_p2p21) as csvfile:
           p2p21 = csvfile.read().splitlines()
           p2p21 = np.asarray(p2p21, dtype=int)
    list_p2p=p2p21

    #Get lists of vertices for surface 1 and surface 2 using TriMesh from pyFM 
    vertices_1, vertices_2 = read_vertices(args.input_mesh1,args.input_mesh2)
    
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

    R,t = optimal_rotation_translation(matrix_A,matrix_B)
    print(t)
    print(R)
if __name__ == "__main__":

    main()
