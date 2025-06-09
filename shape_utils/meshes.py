import pymeshlab as ml
import os
import numpy as np
import pymeshfix
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.distance import squareform

def fix_mesh(mesh_file,resolution, collapse_vertices = False, reconstruct = False):
    ms = ml.MeshSet()
    ms.load_new_mesh(mesh_file)
    if reconstruct:
        new_ms,v,f = reconstruct_mesh(ms)
        ms=new_ms
    if collapse_vertices:
        v,f = reduce_resolution_mesh(ms, resolution)
    else:
        m = ms.current_mesh()
        v = m.vertex_matrix()
        f = m.face_matrix()
    meshfix = pymeshfix.MeshFix(v, f)
    meshfix.repair()
    #output_obj=os.path.join(output_path,'{}.obj'.format(entry))
    #mesh_fixed = trimesh.Trimesh(vertices=meshfix.points, faces=meshfix.faces)

    #mesh_fixed.export(output_obj)
    #remove_first_line(output_obj)

    return meshfix.points, meshfix.faces

def reduce_resolution_mesh(ms, resolution):
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
        #ms = ml.MeshSet()
        #ms.load_new_mesh(mesh_file)
        ms.apply_filter('generate_surface_reconstruction_vcg')
        m = ms.current_mesh()
        return ms, m.vertex_matrix(),m.face_matrix()


def compute_center_of_mass(mesh):
        center = mesh.center_mass
        return center
def compute_min_dist(mesh1_file, mesh2_file):
    """Find the minimum distance between two meshes."""
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
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the index of the first line that starts with 'v' (ignoring leading whitespace)
    start_index = next((i for i, line in enumerate(lines) if line.lstrip().startswith('v')), len(lines))

    # Keep only lines starting from the first 'v' line
    lines = lines[start_index:]

    with open(file_path, 'w') as file:
        file.writelines(lines)