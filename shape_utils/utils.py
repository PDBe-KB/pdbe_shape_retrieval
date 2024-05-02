import logging
import os 
import pandas as pd
from pandas import DataFrame
import numpy as np
import vedo as vp
import pymeshfix 

logger = logging.getLogger()

def save_spectral_descr_to_csv(data, output_file):
    """Dump WKS or HKS descriptors into a output file.                                                                                                                   
        Args:
            data (dict): The data (for spectral descriptors) to be dumped.                                                                        
            output_file (str): The path to the output file.                                                           
                                                                                             
        Raises:                                                                                                        
            ValueError: If `data` is empty.
    """
    if len(data) != 0 :

        try:
            
            descr_data=tuple(map(tuple, data))
            df = pd.DataFrame(descr_data)             
            df.to_csv(output_file, index=False,header=None)

            #return df 


        except Exception as e:
            logging.error(
                "Invalid data frame for wks descriptors: probably wrong fields in the data "
            )
            logging.error(e)
    else:
        logging.info(f"No data found to save")

        return None 

def save_list_to_csv(data, output_file):
    """Dump parameters list into an output file.                                                                                                                   
        Args:
            data (dict): The data for list of parameters to be dumped.                                                                        
            output_file (str): The path to the output file.                                                           
                                                                                             
        Raises:                                                                                                        
            ValueError: If `data` is empty.
    """
    if len(data) != 0 :

        try:
            df = pd.DataFrame(data)             
            df.to_csv(output_file, index=False,header=None)

        except Exception as e:
            logging.error(
                "Invalid data frame list of paramenters: probably wrong fields in the data "
             )
            logging.error(e)
    else:
        logging.info(f"No data found to save")

        return None

def clean_mesh(mesh_file):
    vedo_mesh = vp.load(mesh_file)
    v, f = vedo_mesh.points(), np.asarray(vedo_mesh.faces())
    meshfix = pymeshfix.MeshFix(v, f)
    meshfix.repair()
    return vp.Mesh([meshfix.v, meshfix.f])
