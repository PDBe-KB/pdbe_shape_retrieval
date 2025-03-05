import numpy as np
import argparse
import logging
from shape_utils.pyFM_pdbe.mesh import TriMesh
from shape_utils.spectral_descr import calculate_descriptors
from shape_utils.predict_similarity import calculate_similarity_score
from shape_utils.functional_maps import calculate_functional_maps
from shape_utils.pyFM_pdbe import functional 
from shape_utils.utils import save_data_to_csv, save_list_to_csv, find_minimum_distance_meshes
from shape_utils.zernike_descr import get_inv, plytoobj, predict_similarity
from shape_utils.similarity_scores import calculate_geodesic_norm_score
import pandas as pd
import matplotlib.pyplot as plt
import os 
import multiprocessing 

import cProfile
import pstats
import io


def main():
    def_no_cpu = min(min(8, multiprocessing.cpu_count()), 8)
    #print(multiprocessing.cpu_count())
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
        "--pdb_ids",
        nargs="+",
        help="List of two pdb_ids",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output files (descriptors)",
        required=True,
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
        default=1,
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
    

    if args.descr =='WKS':
        parameter_descr = 'energy'
    if args.descr == 'HKS':
        parameter_descr = 'time'

    pdb_id_1 = args.pdb_ids[0]
    pdb_id_2 = args.pdb_ids[1]

    if args.min_dist_mesh :
       mesh1 = TriMesh(args.input_mesh1, area_normalize=False, center=False)
       mesh2 = TriMesh(args.input_mesh2, area_normalize=False, center=False) 
       min_distance = find_minimum_distance_meshes(mesh1,mesh2)
       print('Minimum distance is:',min_distance)
    if not args.no_shape_retrieval:

        if args.descr =='WKS'or args.descr == 'HKS':

            mesh1 = TriMesh(args.input_mesh1, area_normalize=True, center=False)
            mesh2 = TriMesh(args.input_mesh2, area_normalize=True, center=False)

            #ouput files with descriptors and parameters list for structure 1 and structure 2

            param_list_file = "{}_{}_{}_list.csv".format(parameter_descr,args.descr,pdb_id_1,args.descr,pdb_id_2)
            descr1_file = "{}_descr_{}.csv".format(args.descr,pdb_id_1)
            descr2_file = "{}_descr_{}.csv".format(args.descr,pdb_id_2)

            output_file_1 = os.path.join(args.output,descr1_file)
            output_file_2 = os.path.join(args.output,descr2_file)
            output_file_3 = os.path.join(args.output,param_list_file)

            logging.info(f"Calculating {args.descr} descriptors for structures {pdb_id_1} and {pdb_id_2}")

            model = functional.FunctionalMapping(mesh1,mesh2)

            if not os.path.exists(output_file_1) or not os.path.exists(output_file_2):
            
                descr1,descr2,paramlist = calculate_descriptors(model,args.neigvecs,args.n_ev,args.ndescr,args.step,args.landmarks,args.output,args.descr)
        
                data1 = np.array(descr1)
                data2 = np.array(descr2)
                data3 = np.array(paramlist)
        
                #save descriptors
                save_data_to_csv(data1,output_file_1)
                save_data_to_csv(data2,output_file_2)

                #save parameters list
                save_list_to_csv(data3,output_file_3)


            FM_file = "{}_{}_FM.csv".format(pdb_id_1,pdb_id_2)
            output_FM = os.path.join(args.output,FM_file)
        
            p2p21_file = "{}_{}_p2p21.csv".format(pdb_id_1,pdb_id_2)
            output_p2p21 = os.path.join(args.output, p2p21_file)

            #compute correspondance matrix, shape difference matrix and p2p21         
        
            if not os.path.exists(output_FM) or not os.path.exists(output_p2p21):

                p2p21, FM = calculate_functional_maps(model,args.n_cpus,refine = args.refine)
        
                score_geodesic_norm_eigenvalues = calculate_geodesic_norm_score(FM)
        

                #save descriptors
                save_data_to_csv(FM,output_FM)
                save_list_to_csv(p2p21,output_p2p21)

                print('Disimilarity score is:',score_geodesic_norm_eigenvalues)
        
        elif args.descr =='3DZD':

            if not "map2zernike" and not os.path.isfile(args.map2zernike_binary):
            
                raise Exception(f"map2zernike binary not found or path not provided: {args.map2zernike_binary}")

            if not "obj2grid" and not os.path.isfile(args.obj2grid_binary):

                raise Exception(f"obj2grid binary not found or path to binary not provided: {args.obj2grid_binary}")

            try:

                obj_file_1 = plytoobj(args.input_mesh1,args.output)
                obj_file_2 = plytoobj(args.input_mesh2,args.output)
                get_inv(obj_file_1,args.pdb_ids[0],args.map2zernike_binary, args.obj2grid_binary,args.output)
                get_inv(obj_file_2,args.pdb_ids[1],args.map2zernike_binary, args.obj2grid_binary,args.output)
                #calculate_similarity_score(args.models_zernike,args.output,atom_type = 'fullatom')
                #predict_similarity(args.output,args.output,args.models_zernike)
            except Exception as e:
                logging.error(
                    "something went wrong, probably map2zernike or obj2grid binaries not working properly  "
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
    
