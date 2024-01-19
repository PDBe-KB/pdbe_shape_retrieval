import numpy as np
import argparse
import logging
#from shape_utils import pyFM_pdbe
from shape_utils.pyhks import trimesh, hks
from shape_utils.pyFM_pdbe.mesh import TriMesh
from shape_utils.spectral_descr import calculate_descriptors, distance_WKS, saveWKSColors
from shape_utils.functional_maps import calculate_functional_maps, compute_shape_difference
from shape_utils.pyFM_pdbe import functional 
from scipy import sparse
from scipy.sparse.linalg import lsqr, cg, eigsh
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
        "-o",
        "--output",
        help="Path to output file with L1 distances between wks descriptors of two 3D objects",
        required=True,
    )

    parser.add_argument(
        "--input_mesh2",
        help="Input triangulated mesh 2",
        required=True,
    )
    parser.add_argument(
        "--neigvecs",
        type = int,
        required=False,
        help="No. of eigenvalues/eigenvectors to process (>100). A minimum of neigvecs=100 will be automaticaly set",
        default=200
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
        help="type of descriptor to calculate",
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

    if args.descr =='WKS'or args.descr == 'HKS':
        descr1,descr2,paramlist = calculate_descriptors(model,args.neigvecs,args.ndescr,args.step,args.landmarks,args.output,args.descr)
    else :
        print('descriptor not yet implemented')
    
    descr1_file = "signatures_1.off"
    descr2_file = "signatures_2.off"
    output_file_1 = os.path.join(args.output,descr1_file)
    output_file_2 = os.path.join(args.output,descr2_file)
    saveWKSColors(output_file_1, mesh1.vertlist, descr1[:, 0], mesh1.facelist)
    saveWKSColors(output_file_2, mesh2.vertlist, descr2[:, 0], mesh2.facelist)
    calculate_functional_maps(model,args.n_cpus,refine='zoomout')
    #compute_shape_difference(model)
    #distance_WKS(descr1,descr2,args.output)



if __name__ == "__main__":
    main()
    
    
