import pyvol
from pyvol import identify
import argparse
import logging
import os

def create_pyvol_config(path_setup, output_data, pdb_file):
    """                                                                                                               Function creates a configuration file to run pyvol              
    :param path_config_file: path to setup configuration file
    :param output_dir : path to outputs                                                                  
    :return : None

    """ 

    outputname = os.path.join(path_setup, "config.cfg")
    pdbfile=os.path.join(path_setup, pdb_file)
    with open(os.path.join(path_setup, "config_tmp")) as infile:
        with open(outputname, "w") as outfile:
            for line in infile:
                line = line.replace("pdb_file", pdb_file).replace("path_output",output_data)
                outfile.write(line)
    return outputname

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-i",
        "--input_list_pdbs",
        help="Input list of MD snapshots in pdb format",
        required=True,
    )
    parser.add_argument(
        "--pdb_entry",
        help="pdb entry",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        help="Output path to save results",
        required=True,
    )
    parser.add_argument(
        "--prefix",
        help="prefix for the name of output files",
        required=False,
        default= 'result'
    )
    args = parser.parse_args()
    prefix_label = args.prefix
    #pdb_file = args.input_pdb_file
    pdb_files = open(args.input_list_pdbs)
    pdb_files_list = pdb_files.read().splitlines()
    n=0
    for pdb_file in pdb_files_list:
        out_name = "{}_{}".format(args.pdb_entry,n)
        out_dir = os.path.join(args.out_dir,out_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        pockets, opts = identify.pocket(prot_file=pdb_file, lig_file=None, min_rad=1.4, max_rad=3.4, 
                            constrain_radii=True, mode='all', coordinates=None, residue=None,                         resid=None, lig_excl_rad=None, lig_incl_rad=None, min_volume=200, 
                            subdivide=False, max_clusters=None, min_subpocket_rad=1.7, 
                            max_subpocket_rad=3.4, min_subpocket_surf_rad=1.0, radial_sampling=0.1, 
                            inclusion_radius_buffer=1.0, min_cluster_size=50, project_dir=out_dir, 
                            output_dir= out_dir, prefix=out_name, logger_stream_level='DEBUG', 
                        logger_file_level='INFO', display_mode='solid', alpha=1.0)
        print(out_name,len(pockets))
        n=n+1
        
        #config_file = create_pyvol_config(config_dir, output_dir, pdb_file)
        

if "__main__" in __name__:
    main()
