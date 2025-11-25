import pymeshlab as ml
import os
import numpy as np
import pymeshfix
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.distance import squareform

def fix_mesh(mesh_file,resolution=0.5, collapse_vertices = False, reconstruct = False):
    """
    Repair and optionally simplify a 3D mesh using MeshLab (pymeshlab) and PyMeshFix.

    This function loads a mesh file (.obj or .off), optionally reconstructs it,
    repairs common defects using PyMeshFix, and optionally collapses vertices
    to reduce mesh resolution.

    Args:
        mesh_file (str):
            Path to the mesh file to be processed. Supported formats are typically
            `.obj` and `.off`, depending on MeshLab's loader.

        resolution (int or float):
            Target fraction reduction (or simplification parameter) passed to
            `reduce_resolution_mesh()` to collapse vertices. 

        collapse_vertices (bool, optional):
            If True, the function collapses/simplifies the mesh after repairing it.
            Defaults to False.

        reconstruct (bool, optional):
            If True, attempts to reconstruct the mesh using `reconstruct_mesh()`
            before performing repairs. Useful for badly conditioned or incomplete meshes.
            Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            A tuple `(v, f)` where:
            - `v` is an (N, 3) array of vertex positions.
            - `f` is an (M, 3) array of face indices.

    Notes:
        - PyMeshFix is used to repair self-intersections, holes, and non-manifold edges.
        - MeshLab (pymeshlab) is used for optional reconstruction and resolution reduction.
        - The function does not write output to disk; it returns the processed arrays.
    """
    
    ms = ml.MeshSet()
    ms.load_new_mesh(mesh_file)
    if reconstruct:
        new_ms,v,f = reconstruct_mesh(ms)
        ms=new_ms
    m = ms.current_mesh()
    v = m.vertex_matrix()
    f = m.face_matrix()
    meshfix = pymeshfix.MeshFix(v, f)
    meshfix.repair()
    if collapse_vertices:
        ms.add_mesh(ml.Mesh(meshfix.points, meshfix.faces))
        v,f = reduce_resolution_mesh(ms, resolution)
    else:
        v = meshfix.points
        f = meshfix.faces

    return v, f

def reduce_resolution_mesh(ms, resolution):
    """
    Reduce the resolution of the current mesh in a MeshSet using quadratic edge collapse of vertices.

    This function simplifies the mesh by collapsing edges based on the quadric
    error metric. The number of target faces is computed as a fraction of the
    mesh’s current vertex count, controlled by the `resolution` parameter.

    Args:
        ms (pymeshlab.MeshSet):
            A MeshSet object containing the mesh to be simplified. The current
            active mesh is used for decimation.

        resolution (float):
            Fraction of the current vertex count used to determine the target
            number of faces. Typically:
             0 < resolution < 1  → reduce to a fraction of the original size
           
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            `(v, f)` where:
            - `v` is the simplified vertex matrix (N, 3).
            - `f` is the simplified face matrix (M, 3).

    Notes:
        - Uses MeshLab's `meshing_decimation_quadric_edge_collapse` filter.
        - Normals, boundaries, and topology are preserved during simplification.
    """
    m = ms.current_mesh()
    TARGET = int(resolution*m.vertex_number())
    ms.apply_filter('meshing_decimation_quadric_edge_collapse',
                    targetfacenum=TARGET, preservenormal=True,
                    preserveboundary=True,preservetopology=True)
    m = ms.current_mesh()
    v = m.vertex_matrix()
    f = m.face_matrix()
    return v,f

def reconstruct_mesh(ms):
    """
    Reconstruct a surface mesh using MeshLab's VCG surface reconstruction filter. This function applies the `generate_surface_reconstruction_vcg` filter to the
    current mesh in the provided MeshSet, producing a reconstructed (often watertight)
    surface.

    Args:
    ms (pymeshlab.MeshSet):
        A MeshSet containing the mesh to reconstruct. The reconstruction is
        applied to the current active mesh.

    Returns:
        Tuple[pymeshlab.MeshSet, np.ndarray, np.ndarray]:
            A tuple `(ms, v, f)` where:
            - `ms` is the MeshSet after reconstruction,
            - `v` is the reconstructed vertex matrix (N, 3),
            - `f` is the reconstructed face matrix (M, 3).

    Raises:
        ValueError:
            If the MeshSet has no current mesh to reconstruct.
    """
    ms.apply_filter('generate_surface_reconstruction_vcg')
    m = ms.current_mesh()
    return ms, m.vertex_matrix(),m.face_matrix()


def compute_center_of_mass(mesh):
    center = mesh.center_mass
    return center
def compute_min_dist(mesh1_file, mesh2_file):
    """
    Compute minimum distance between two meshes
    
    Args:
    mesh1_file:
        Path to the first mesh file to be processed. Supported formats are typically
        `.obj` and `.off`.
    mesh1_file:
        Path to the second mesh file to be processed. Supported formats are typically
        `.obj` and `.off`.    
    """
    mesh1=trimesh.load_mesh(mesh1_file)
    mesh2 = trimesh.load_mesh(mesh2_file)
    # Get the vertices of each mesh
    vertices1 = mesh1.vertices
    vertices2 = mesh2.vertices
    
    # Use a KDTree for efficient nearest neighbor search
    tree = KDTree(vertices2)
    distances, _ = tree.query(vertices1)
    # Return the minimum distance
    return np.min(distances)

def compute_centers_dist(mesh1_file,mesh2_file):
    """
    Compute the Euclidean distance between the centers of mass of two meshes.
    This function loads two mesh files using `trimesh`, computes their centers
    of mass (volume-based when possible), and returns the Euclidean distance
    between those centers.

    Args:
        mesh1_file (str):
            Path to the first mesh file.

        mesh2_file (str):
            Path to the second mesh file.

    Returns:
        float:
            The Euclidean distance between the centers of mass of the two meshes.

    Raises:
        ValueError:
            If either mesh fails to load or is empty.
        ImportError:
            If `trimesh` or `numpy` are not installed.

    """
    # Load two meshes
    mesh1 = trimesh.load_mesh(mesh1_file)
    mesh2 = trimesh.load_mesh(mesh2_file)

    # Compute center of mass (volume-based if possible)
    center1 = mesh1.center_mass 
    center2 = mesh2.center_mass 

    # Calculate Euclidean distance between centers
    distance = np.linalg.norm(center1 - center2)
    return distance
def remove_until_vertex(file_path):
    """
    Remove all lines in a mesh file until the first vertex declaration.

    This function scans the file and discards all content before the first line
    that begins with `'v''. It is primarily useful for cleaning corrupted or non-standard OBJ files 
    that cannot be read for Zernike descriptors. 
    Args:
        file_path (str):
            Path to the mesh file to be cleaned. The file is modified in place.

    Returns:
        None

    Raises:
        FileNotFoundError:
            If the specified file does not exist.

        ValueError:
            If the file contains no vertex lines starting with `'v'`.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the index of the first line that starts with 'v' (ignoring leading whitespace)
    start_index = next((i for i, line in enumerate(lines) if line.lstrip().startswith('v')), len(lines))

    # Keep only lines starting from the first 'v' line
    lines = lines[start_index:]

    with open(file_path, 'w') as file:
        file.writelines(lines)