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
from shape_utils.zernike_descr import get_inv, plytoobj, predict_similarity
from shape_utils.similarity_scores import calculate_geodesic_norm_score

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
        "--input_mesh1",
        help="Input triangulated mesh 1",
        required=True,
    )

    parser.add_argument(
        "--input_mesh2",
        help="Input triangulated mesh 2",
        required=True,
    )
    parser.add_argument(
        "--entry_ids",
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
        "--fix_meshes",
        action="store_true", 
        help="Preprocess meshes to be well-conditioned to compute descriptors",
        required=False,
    )

    parser.add_argument(
        "--collapse_vertices",
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
        "--reconstruct_mesh", 
        action="store_true", 
        default = False,
        help="Reconstruct mesh to create a well-conditioned mesh",
        required=False,
    )


    parser.add_argument(
        "--map2zernike_binary",
        help="path to map2zernike binary",
        default = 'map2zernike',
    )
    parser.add_argument(
        "--obj2grid_binary",
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
        "--n_ev",
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
        help="Type of descriptor to calculate: WKS,HKS,Zernike",
        default='WKS',
    )
    
    parser.add_argument(
        "--n_cpus",
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
        "--min_dist_mesh", 
        action="store_true", 
        help="Calculate minimum distance between the two meshes",
        required=False,
    )
    parser.add_argument(
        "--no_shape_retrieval", 
        action="store_true", 
        help="switch off shape retrieval calculation",
        required=False,
    )

    parser.add_argument(
        "--models_zernike",
        type=str,
        help="Path to Best models for neural network and similarity scores of Zernike descriptors",
    )
  
    args = parser.parse_args()
    

    #if args.descr =='WKS':
    #    parameter_descr = 'energy'
    #if args.descr == 'HKS':
    #    parameter_descr = 'time'

    entry_id_1 = args.entry_ids[0]
    entry_id_2 = args.entry_ids[1]
    mesh1_file = args.input_mesh1
    mesh2_file = args.input_mesh2
    resolution = args.resolution
    reconstruct = args.reconstruct_mesh

    if reconstruct and not args.fix_meshes:
        logging.error(
                    "--recontruct_mesh needs to be used with --fix_meshes flag  "
                )
    if args.collapse_vertices and not args.fix_meshes:
        logging.error(
                    "--collapse_vertices needs to be used with --fix_meshes flag  "
                )

    if args.fix_meshes :
        
        v_1,f_1=fix_mesh(mesh1_file,resolution,args.collapse_vertices,reconstruct)
        v_2,f_2=fix_mesh(mesh2_file,resolution,args.collapse_vertices,reconstruct) 
        
    if args.min_dist_mesh :
       if args.fix_meshes:
             mesh1 = mesh.TriMesh(v_1, f_1)
             mesh2 = mesh.TriMesh(v_2, f_2)
       else:
            mesh1 = mesh.TriMesh(args.input_mesh1, area_normalize=False, center=False)
            mesh2 = mesh.TriMesh(args.input_mesh2, area_normalize=False, center=False) 
       
       min_distance = find_minimum_distance_meshes(mesh1,mesh2)
       print('Minimum distance is:',min_distance)
    
    if not args.no_shape_retrieval:

        if args.descr =='WKS'or args.descr == 'HKS':
            if args.fix_meshes:
                mesh1 = mesh.TriMesh(v_1,f_1, area_normalize=True, center=False)
                mesh2 = mesh.TriMesh(v_2,f_2, area_normalize=True, center=False)
            else:    
                mesh1 = mesh.TriMesh(args.input_mesh1, area_normalize=True, center=False)
                mesh2 = mesh.TriMesh(args.input_mesh2, area_normalize=True, center=False)

            #ouput files with descriptors and parameters list for structure 1 and structure 2

            #param_list_file = "{}_{}_{}_list.csv".format(parameter_descr,args.descr,entry_id_1,args.descr,entry_id_2)
            descr1_file = "{}_descr_{}.csv".format(args.descr,entry_id_1)
            descr2_file = "{}_descr_{}.csv".format(args.descr,entry_id_2)

            output_file_1 = os.path.join(args.output,descr1_file)
            output_file_2 = os.path.join(args.output,descr2_file)
            #output_file_3 = os.path.join(args.output,param_list_file)

            logging.info(f"Calculating {args.descr} descriptors for structures {entry_id_1} and {entry_id_2}")

            model = functional.FunctionalMapping(mesh1,mesh2)

            if not os.path.exists(output_file_1) or not os.path.exists(output_file_2):
            
                descr1,descr2 = calculate_descriptors(model,args.neigvecs,args.n_ev,args.ndescr,args.step,args.landmarks,args.output,args.descr)
                data1 = np.array(descr1)
                data2 = np.array(descr2)
                #data3 = np.array(paramlist)
        
                #save descriptors
                save_data_to_csv(data1,output_file_1)
                save_data_to_csv(data2,output_file_2)

                #save parameters list
                #save_list_to_csv(data3,output_file_3)


            FM_file = "{}_{}_FM.csv".format(entry_id_1,entry_id_2)
            output_FM = os.path.join(args.output,FM_file)
        
            p2p21_file = "{}_{}_p2p21.csv".format(entry_id_1,entry_id_2)
            output_p2p21 = os.path.join(args.output, p2p21_file)

            #compute correspondance matrix, shape difference matrix and p2p21         
            if os.path.exists(output_FM) and os.path.exists(output_p2p21):
                with open(output_FM) as FMfile:
                    FM = list(csv.reader(FMfile))
                    FM = np.asarray(FM, dtype=float)
                score_geodesic_norm_eigenvalues = calculate_geodesic_norm_score(FM)
                print('Disimilarity score is:',score_geodesic_norm_eigenvalues)

            if not os.path.exists(output_FM) or not os.path.exists(output_p2p21):

                p2p21, FM = calculate_functional_maps(model,args.n_cpus,refine = args.refine)
        
                score_geodesic_norm_eigenvalues = calculate_geodesic_norm_score(FM)
            
                #save correspondence matrix and point to point map
                save_data_to_csv(FM,output_FM)
                save_list_to_csv(p2p21,output_p2p21)

                print('Disimilarity score is:',score_geodesic_norm_eigenvalues)
           
        elif args.descr =='3DZD':

            if not "map2zernike" and not os.path.isfile(args.map2zernike_binary):
            
                raise Exception(f"map2zernike binary not found or path not provided: {args.map2zernike_binary}")

            if not "obj2grid" and not os.path.isfile(args.obj2grid_binary):

                raise Exception(f"obj2grid binary not found or path to binary not provided: {args.obj2grid_binary}")

            try:
                if args.fix_meshes:
                    mesh_1 = trimesh.Trimesh(vertices=v_1, faces=f_1)
                    mesh_2 = trimesh.Trimesh(vertices=v_2, faces=f_2)
                    output1_obj=os.path.join(args.output,'{}_fixed.obj'.format(entry_id_1))
                    output2_obj=os.path.join(args.output,'{}_fixed.obj'.format(entry_id_2))
                    mesh_1.export(output1_obj)
                    mesh_1.export(output2_obj)
                    remove_until_vertex(output1_obj)
                    remove_until_vertex(output2_obj)
                    get_inv(output1_obj,args.entry_ids[0],args.map2zernike_binary, args.obj2grid_binary,args.output)
                    get_inv(output2_obj,args.entry_ids[1],args.map2zernike_binary, args.obj2grid_binary,args.output)
                
                else:
                    _, ext1 = os.path.splitext(args.input_mesh1)
                    _, ext2 = os.path.splitext(args.input_mesh2)
                    if ext1.lower() != '.obj' or ext2.lower() != '.obj':
                        raise Exception("Zernike descriptors take '.obj' files as input, try option --fix_meshes to compute them"
                    )
                    remove_until_vertex(args.input_mesh1)
                    remove_until_vertex(args.input_mesh2)
            
                    get_inv(args.input_mesh1,args.entry_ids[0],args.map2zernike_binary, args.obj2grid_binary,args.output)
                    get_inv(args.input_mesh2,args.entry_ids[1],args.map2zernike_binary, args.obj2grid_binary,args.output)
                
            except Exception as e:
                logging.error(
                    "something went wrong, check that mesh file is well-conditioned and map2zernike or obj2grid binaries are working properly  "
                )
                logging.error(e)
            
        else :
            print('Descriptor not yet implemented')
    

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
    
