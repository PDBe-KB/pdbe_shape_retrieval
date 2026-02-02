import numpy as np
import csv
import argparse
import logging
from pyFM import mesh
from pyFM import functional
import trimesh

from shape_utils.spectral_descr import calculate_descriptors
from shape_utils.functional_maps import calculate_functional_maps
from shape_utils.meshes import fix_mesh, remove_until_vertex
from shape_utils.utils import save_data_to_csv, save_list_to_csv, find_minimum_distance_meshes
from shape_utils.zernike_descr import get_inv
from shape_utils.similarity_scores import calculate_geodesic_norm_score,predict_similarity_zernike

#import pandas as pd
#import matplotlib.pyplot as plt
import os 
import multiprocessing 

import cProfile
import pstats
import io


def main():
    def_no_cpu = multiprocessing.cpu_count()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mesh1",
        help="Input path to triangulated mesh file 1",
        required=True,
    )

    parser.add_argument(
        "--mesh2",
        help="Input path to triangulated mesh file 2",
        required=True,
    )
    parser.add_argument(
        "--entry-ids",
        nargs="+",
        help="List of two entry ids",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output files (descriptors)",
        required=True,
    )

    parser.add_argument(
        "--fix-meshes",
        action="store_true", 
        help="Preprocess meshes to be well-conditioned to compute descriptors",
        required=False,
    )

    parser.add_argument(
        "--collapse-vertices",
        action=argparse.BooleanOptionalAction,
        default = False,
        help="Use decimation quadric edgecollapse to reduce resolution of mesh",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        required=False,
        default = 0.5 ,
        help="Factor to collapse No. of vertices, e.g 0.5 will reduce the vertices to ~half",
    )

    parser.add_argument(
        "--reconstruct-mesh", 
        action="store_true", 
        default = False,
        help="Reconstruct mesh to create a well-conditioned mesh",
        required=False,
    )


    parser.add_argument(
        "--zernike-binary",
        help="path to map2zernike binary",
        default = 'map2zernike',
    )
    parser.add_argument(
        "--obj2grid-binary",
        help="path to obj2grid binary (needed for Zernike descriptors)",
        default = "obj2grid",
    )
    parser.add_argument(
        "--neigvecs",
        type = int,
        required=False,
        help="No. of eigenvalues/eigenvectors to process (>100). A minimum of neigvecs=200 will be automaticaly set",
        default=200
    )
    parser.add_argument(
        "--nev",
        type = int,
        required=False,
        help="The least number of Laplacian eigenvalues to consider for functional map ",
        default = 50
    )
    parser.add_argument(
	"--ndescr",
	help="No. of descriptors to process, if no value is given, 50 descriptors will be used for spectral calculations",
	default= 100
    )
    parser.add_argument(
        "--step",
        help="Subsample step in order not to use too many spectral descriptors",
        default=1
    )
    parser.add_argument(
        "--landmarks",
        help="Input indices of landmarks for spectral descriptors",
        default=None,
    )    
    parser.add_argument(
        "--descr",
        type=str,
        help="Type of descriptor to calculate: WKS,HKS,3DZD",
        default='WKS',
    )
    
    parser.add_argument(
        "--ncpus",
        type=int,
        default=def_no_cpu,
        help="Number of threads to be used for this calculation.",
        required=False,
    )

    parser.add_argument(
        "--refine",
        type=str,
        default = None,        
        help="Use refining method for calculation of fuctional maps: icp,zoomout",
        required=False,
    )

    parser.add_argument(
        "--mindist", 
        action="store_true", 
        help="Calculate minimum distance between the two meshes",
        required=False,
    )
    parser.add_argument(
        "--no-shape", 
        action="store_true", 
        help="switch off shape retrieval calculation",
        required=False,
    )

  
    args = parser.parse_args()
    

    entry_id_1 = args.entry_ids[0]
    entry_id_2 = args.entry_ids[1]
    mesh1_file = args.input_mesh1
    mesh2_file = args.input_mesh2
    resolution = args.resolution
    reconstruct = args.reconstruct_mesh

    if reconstruct and not args.fix_meshes:
        logging.error(
                    "--reconstruct_mesh must be used together with the --fix_meshes flag."
                )
    if args.collapse_vertices and not args.fix_meshes:
        logging.error(
                    "--collapse_vertices must be used with --fix_meshes flag  "
                )
    #Repair common mesh defects using PyMeshFix,
    #and optionally collapses vertices to reduce mesh resolution
    if args.fix_meshes :
        
        v_1,f_1=fix_mesh(mesh1_file,resolution,args.collapse_vertices,reconstruct)
        v_2,f_2=fix_mesh(mesh2_file,resolution,args.collapse_vertices,reconstruct) 
        logging.info("Repairing meshes: {} and {}".format(mesh1_file,mesh2_file))
    #Compute minimum distance between two meshes   
    if args.min_dist_mesh :
       if args.fix_meshes:
            mesh1 = mesh.TriMesh(v_1, f_1)
            mesh2 = mesh.TriMesh(v_2, f_2)
       else:
            mesh1 = mesh.TriMesh(args.input_mesh1, area_normalize=False, center=False)
            mesh2 = mesh.TriMesh(args.input_mesh2, area_normalize=False, center=False) 
       
       min_distance = find_minimum_distance_meshes(mesh1,mesh2)
       logging.info("Minimum distance is:{}".format(min_distance))
       print('Minimum distance is:',min_distance)
    
    if not args.no_shape_retrieval:

        if args.descr =='WKS'or args.descr == 'HKS':
            #Computing Spectral descriptors with pyFM on area normalized meshes
            if args.fix_meshes:
                mesh1 = mesh.TriMesh(v_1,f_1, area_normalize=True, center=False)
                mesh2 = mesh.TriMesh(v_2,f_2, area_normalize=True, center=False)
            else:    
                mesh1 = mesh.TriMesh(args.input_mesh1, area_normalize=True, center=False)
                mesh2 = mesh.TriMesh(args.input_mesh2, area_normalize=True, center=False)

            
            #Ouput files with descriptors

            descr1_file = "{}_descr_{}.csv".format(args.descr,entry_id_1)
            descr2_file = "{}_descr_{}.csv".format(args.descr,entry_id_2)

            output_file_1 = os.path.join(args.output,descr1_file)
            output_file_2 = os.path.join(args.output,descr2_file)
            
            #Setting model for WKS/Hks descriptors in pyFM
            model = functional.FunctionalMapping(mesh1,mesh2)

            if not os.path.exists(output_file_1) or not os.path.exists(output_file_2):
                logging.info(f"Calculating {args.descr} descriptors for structures {entry_id_1} and {entry_id_2}")
                
                descr1,descr2 = calculate_descriptors(model,args.neigvecs,args.n_ev,args.ndescr,args.step,args.landmarks,args.output,args.descr)
                data1 = np.array(descr1)
                data2 = np.array(descr2)
                
                #save descriptors
                save_data_to_csv(data1,output_file_1)
                save_data_to_csv(data2,output_file_2)

            #Ouput files with functional maps and point to point maps

            FM_file = "{}_{}_FM.csv".format(entry_id_1,entry_id_2)
            output_FM = os.path.join(args.output,FM_file)
        
            p2p21_file = "{}_{}_p2p21.csv".format(entry_id_1,entry_id_2)
            output_p2p21 = os.path.join(args.output, p2p21_file)

            #Computing correspondence matrix  and point to point map p2p21 (if not already calculated)
            # Point to point map defined from mesh2 to mesh1
                    
            if os.path.exists(output_FM) and os.path.exists(output_p2p21):
                with open(output_FM) as FMfile:
                    FM = list(csv.reader(FMfile))
                    FM = np.asarray(FM, dtype=float)
                #Calculating similarity score
                score_geodesic_norm_eigenvalues = calculate_geodesic_norm_score(FM)
                logging.info("Shspe Disimilarity score is::{}".format(score_geodesic_norm_eigenvalues))
                print('Disimilarity score is:',score_geodesic_norm_eigenvalues)
                
            if not os.path.exists(output_FM) or not os.path.exists(output_p2p21):

                p2p21, FM = calculate_functional_maps(model,args.n_cpus,refine = args.refine)
        
                score_geodesic_norm_eigenvalues = calculate_geodesic_norm_score(FM)
                
                #save correspondence matrix and point to point map
                save_data_to_csv(FM,output_FM)
                save_list_to_csv(p2p21,output_p2p21)

                print('Disimilarity_score is:',score_geodesic_norm_eigenvalues)
        
        #Computing Zernike descriptors  
        elif args.descr =='3DZD':
            #Check paths to 3DZD binaries required to calculate descriptors
            if not "map2zernike" and not os.path.isfile(args.map2zernike_binary):
            
                raise Exception(f"map2zernike binary not found or path not provided: {args.map2zernike_binary}")

            if not "obj2grid" and not os.path.isfile(args.obj2grid_binary):

                raise Exception(f"obj2grid binary not found or path to binary not provided: {args.obj2grid_binary}")

            try:
                if args.fix_meshes:
                    mesh_1 = trimesh.Trimesh(vertices=v_1, faces=f_1)
                    mesh_2 = trimesh.Trimesh(vertices=v_2, faces=f_2)
                    #Save fixed meshes
                    output1_obj=os.path.join(args.output,'{}_fixed.obj'.format(entry_id_1))
                    output2_obj=os.path.join(args.output,'{}_fixed.obj'.format(entry_id_2))
                    mesh_1.export(output1_obj)
                    mesh_2.export(output2_obj)

                    #Remove headers in OBJ mesh files to run 3DZD binaries 
                    remove_until_vertex(output1_obj)
                    remove_until_vertex(output2_obj)
                    get_inv(output1_obj,args.entry_ids[0],args.map2zernike_binary, args.obj2grid_binary,args.output)
                    get_inv(output2_obj,args.entry_ids[1],args.map2zernike_binary, args.obj2grid_binary,args.output)
                
                else:
                    _, ext1 = os.path.splitext(args.input_mesh1)
                    _, ext2 = os.path.splitext(args.input_mesh2)

                    if ext1.lower() != '.obj' or ext2.lower() != '.obj':
                        raise Exception("Zernike descriptors take '.obj' files as input.")
                    
                    #Remove headers in OBJ mesh files to run 3DZD binaries 
                    remove_until_vertex(args.input_mesh1)
                    remove_until_vertex(args.input_mesh2)
            
                    get_inv(args.input_mesh1,args.entry_ids[0],args.map2zernike_binary, args.obj2grid_binary,args.output)
                    get_inv(args.input_mesh2,args.entry_ids[1],args.map2zernike_binary, args.obj2grid_binary,args.output)
                
                #predict_similarity_zernike(args.output,args.output)

            except Exception as e:
                logging.error(
                    "something went wrong, check that mesh file is well-conditioned with valid header and map2zernike or obj2grid binaries are working properly  "
                )
                logging.error(e)
            
        else :
            print('Descriptor type not yet implemented')
    

if __name__ == "__main__":

    pr = cProfile.Profile()
    pr.enable()

    main()

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()

    with open('test.txt', 'w+') as f:
        f.write(s.getvalue())
    #cProfile.run('main()')
    
