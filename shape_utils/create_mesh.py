import logging
import os
import subprocess
import tempfile
from time import time
import vedo as vp
from vedo import Mesh as vMesh
import pymeshfix
import numpy as np 

def clean_mesh(mesh_file,id_entry, output_path):
    """  cleans and fixes triangle mesh file (.obj)      

    Expects meshio command-line tool installed (https://pypi.org/project/meshio/) and pymeshfix (https://pymeshfix.pyvista.org)                                                                                                                                              
                                                                                                                       
    Args:                                                                                                              
        mesh_file (str): Path to mesh file                                                                       
        id_entry  (str): ID for pdb entry and chain id : e.g. '6nrx-A' 
        output_path (str) : Path to output directory in which the clean meshes are saved                                                
                                                                                                                       
    Returns:                                                                                                           
        vertices and faces of the fixed mesh
    """
    vedo_mesh = vp.load(mesh_file)
    v, f = vedo_mesh.points(), np.asarray(vedo_mesh.faces())
    meshfix = pymeshfix.MeshFix(v, f)
    meshfix.repair()
    with tempfile.TemporaryDirectory() as temp_dir:
        output_ply=os.path.join(temp_dir,'{}.ply'.format(id_entry))
        output_off=os.path.join(output_path,'{}.off'.format(id_entry))
        meshfix.save(output_ply)
        convert_cmmd = 'meshio'+' '+'convert'+' '+output_ply+' '+output_off
        ls_cmmd = 'ls'+' '+temp_dir
        os.system(convert_cmmd)
        os.system(convert_cmmd)
        fix_off_file(output_off)

    return vp.Mesh([meshfix.v, meshfix.f])

def fix_off_file(file_off_mesh):
    """  Fixes format and re-writes mesh file (.off) created with meshio (basically removes extra empty line that messes pyFM mesh reader)                                                                                                                                                   
                                                                                                                       
    Args:                                                                                                              
        file_off_mesh (str): Path to mesh file created with meshio (.off file)                                                                   
                                                                                                                       
    Returns:                                                                                                           
       
    """
    try:
        # Read the file into a list of lines                                                                           
        with open(file_off_mesh, 'r') as file:
            lines = file.readlines()

        # Erase the second, third, and fifth lines                                                                     
        lines.pop(1)  # Remove the second line                                                                         
        lines.pop(1)  # Remove the new second line (originally the third line)                                         
        lines.pop(2)  # Remove the fifth line                                                                          

        # Write the modified lines back to the file                                                                    
        with open(file_off_mesh, 'w') as file:
            file.writelines(lines)

    except FileNotFoundError:
        print(f"Error: File '{file_off_mesh}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def create_mesh_molstar(input_list, output_meshes, surf_calc_setup_dir, data_source_dir = None):

    """  Creates triangle mesh with mol* and preprocesses mesh with pymeshfix for shape descriptors calculations       

    Expects Node 20.8.0 installed and surface-calculator installed (https://github.com/midlik/surface-calculator)                                                                                                                                                    
                                                                                                                       
    Args:                                                                                                              
        input_list (str): Path to list file of pdb entries and chains                                                                       
        output_meshes (str) : Path to directory where output meshes *.off will be saved 
        node_setup_dir (str) : Directory containing surface_calculator lib/index.js  
        data_source_dir (str): Path to directory containing input cif files to be used as data source by mol*. (optional)                                                 
                                                                                                                       
    Returns:                                                                                                           

    """
    data_file = open(str(input_list))
    index_js_file = os.path.join(surf_calc_setup_dir,'lib/index.js')
    data_entries_chains=data_file.read().splitlines()
    pdb_ids = []
    ids = []
    for line in data_entries_chains:
        p = line.split(',')
        id_entry = str(p[0]+'-'+p[1])
        pdb_ids.append(p[0])
        ids.append(id_entry)
    with tempfile.TemporaryDirectory() as temp_dir:
        if data_source_dir is not None:
            logging.info(f"Running mol* to compute molecular surfaces using data in {data_source_dir} with ids from list {input_list}")
            cif_data = os.path.join(data_source_dir,'{id}.cif')
            subprocess.run(
                    ['node', index_js_file ,input_list,temp_dir, '--quality', 'medium', '--probe', '1.4','--source','file:///'+cif_data],
                    check=True,
                )
        if data_source_dir is None:
            logging.info(f"Running mol* to compute molecular surfaces using entries in {input_list}")

            subprocess.run(
                    ['node', index_js_file ,input_list,temp_dir, '--quality', 'medium', '--probe', '1.4'],
                    check=True,
                )
            
        for id_entry,pdb_id in zip(ids,pdb_ids):
            
            mesh_molstar_zip = os.path.join(temp_dir,id_entry+'.zip')
            
            obj_dir = os.path.join(temp_dir,id_entry)
            subprocess.run(
                ['unzip', '-n',mesh_molstar_zip,'-d',obj_dir],
                check=True,
                stdout = subprocess.DEVNULL,
            )
            
            mesh_molstar = os.path.join(temp_dir,id_entry,pdb_id+'.obj')
            clean_mesh(mesh_molstar,id_entry,output_meshes)

            
