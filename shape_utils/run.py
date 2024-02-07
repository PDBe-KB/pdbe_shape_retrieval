import numpy as np
import argparse
import logging
from shape_utils.pyhks import trimesh, hks
from shape_utils.pyFM_pdbe.mesh import TriMesh
from shape_utils.spectral_descr import calculate_descriptors
from shape_utils.functional_maps import calculate_functional_maps
from shape_utils.pyFM_pdbe import functional 
from shape_utils.utils import save_spectral_descr_to_csv, save_list_to_csv
from shape_utils.zernike_descr import get_inv
import pandas as pd
import matplotlib.pyplot as plt
import os 
import multiprocessing 

def main():
    def_no_cpu = min(min(8, multiprocessing.cpu_count()), 8)
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
        help="List of pdb_ids",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output file with L1 distances between wks descriptors of two 3D objects",
        required=True,
    )
    parser.add_argument(
        "--map2zernike_binary",
        help="map2zernike binary",
        required=False,
    )
    parser.add_argument(
        "--obj2grid_binary",
        help="obj2grid binary (needed for Zernike descriptors)",
        required=False,
    )
    parser.add_argument(
        "--neigvecs",
        type = int,
        required=False,
        help="No. of eigenvalues/eigenvectors to process (>100). A minimum of neigvecs=100 will be automaticaly set",
        default=200
    )
    parser.add_argument(
        "--n_ev",
        type = int,
        required=False,
        help="The least number of Laplacian eigenvalues to consider for functional map ",
        default=50
    )
    parser.add_argument(
	"--ndescr",
	help="No. of descriptors to process",
	default=100
    )
    parser.add_argument(
        "--step",
        help="Subsample step in order not to use too many descriptors",
        default=2
    )
    parser.add_argument(
        "--landmarks",
        help="Input indices of landmarks",
        default=None,
    )    
    parser.add_argument(
        "--descr",
        type=str,
        help="Type of descriptor to calculate:WKS,HKS,Zernike",
        default='WKS',
    )
    
    parser.add_argument(
        "--t",
        type=float,
        default=5,
        help="Time parameter for the HKS"
    )

    parser.add_argument(
        "--n_cpus",
        type=int,
        default=def_no_cpu,
        help="Number of threads to be used for this calculation.",
        required=False,
    )
  
    args = parser.parse_args()


    #if args.descr == "hks":
    #    if not args.input_mesh2:
    #        (VPos, VColors, ITris) = trimesh.load_off(args.input_mesh1)
    #        neigvecs = min(VPos.shape[0], args.neigvecs)
    #        descr = hks.get_hks(VPos, ITris, neigvecs, np.array([args.t]))
    #        filename = "hks_signatures.off"
    #        output_file = os.path.join(output_path,filename)
    #        hks.saveHKSColors(output_file, VPos, descr[:, 0], ITris)
    
    mesh1 = TriMesh(args.input_mesh1, area_normalize=True, center=False)
    mesh2 = TriMesh(args.input_mesh2, area_normalize=True, center=False)
    model = functional.FunctionalMapping(mesh1,mesh2) 

    if args.descr =='WKS':
        parameter_descr = 'energy'
    if args.descr == 'HKS':
        parameter_descr = 'time'

    descr1_file = "{}_descr_1.csv".format(args.descr)
    descr2_file = "{}_descr_2.csv".format(args.descr)
    
    if args.descr == 'WKS' or args.descr == 'HKS':
        param_list_file = "{}_list.csv".format(parameter_descr)

    if args.descr =='WKS'or args.descr == 'HKS':
        descr1,descr2,paramlist = calculate_descriptors(model,args.neigvecs,args.n_ev,args.ndescr,args.step,args.landmarks,args.output,args.descr)
        
        data1 = np.array(descr1)
        data2 = np.array(descr2)
        data3 = np.array(paramlist)

        output_file_1 = os.path.join(args.output,descr1_file)
        output_file_2 = os.path.join(args.output,descr2_file)
        output_file_3 = os.path.join(args.output,param_list_file)
        

        #save descriptors
        save_spectral_descr_to_csv(data1,output_file_1)
        save_spectral_descr_to_csv(data2,output_file_2)

        #save parameters
        save_list_to_csv(data3,output_file_3)

        #compute shape difference matrix and p2p21                                                                                          
        D_a, D_c, p2p21 = calculate_functional_maps(model,args.n_cpus,refine=None)

        
    elif args.descr =='3DZD':

        if not "map2zernike" and not os.path.isfile(args.map2zernike_binary):
            
            raise Exception(f"map2zernike binary not found or is not a file: {args.map2zernike_binary}")

        if not "obj2grid" and not os.path.isfile(args.obj2grid_binary):

            raise Exception(f"obj2grid binary not found or is not a file: {args.obj2grid_binary}")

        try:
            get_inv(args.input_mesh1,args.pdb_ids[0],args.map2zernike_binary, args.obj2grid_binary,args.output)
            
        except Exception as e:
            logging.error(
                "something went wrong, probably map2zernike or obj2grid binaries not working properly  "
            )
            logging.error(e)
            
    else :
        print('Descriptor not yet implemented')
    

if __name__ == "__main__":
    main()
    
    
