import numpy as np
#import matplotlib.pyplot as plt
#from shape_utils.pyFM_pdbe.mesh import TriMesh
from shape_utils.pyFM_pdbe import functional 
#from shape_utils.pyhks.trimesh import save_off
#import argparse
import logging
import seaborn as sns
#import os 
#import pandas as pd
#from pandas import DataFrame


logger = logging.getLogger()

def visu(vertices):
    min_coord,max_coord = np.min(vertices,axis=0,keepdims=True),np.max(vertices,axis=0,keepdims=True)
    cmap = (vertices-min_coord)/(max_coord-min_coord)
    return cmap

def calculate_functional_maps(model,n_cpus=1, refine='zoomout'):
    """                                                                                                                                                                                
    Calculate functional maps with pyFM code (https://github.com/RobinMagnet/pyFM)                                                                                                                                        
                                                                                                                                                                                       
    Returns functional maps and fitted model   

    Args:               
        mesh1 (list) : array with vertices and faces 
        mesh2 (list) : second array with vertices and faces 
        model (int) : functional maps model pyFM
        refine (str) : Selected method to refine functional map                                                                                                                                                           

    """   
    print('cpus used', n_cpus)
    fit_params = {
        'w_descr': 1e0,
        'w_lap': 1e-2,
        'w_dcomm': 1e-1,
        'w_orient': 0
    }

    model.fit(**fit_params, verbose=True)
    p2p_21 = model.get_p2p(n_jobs=4)
    #cmap1 = visu(mesh1.vertlist); cmap2 = cmap1[p2p_21]

    #refine model using ICP or Zoom
    if refine == 'icp':
        model.icp_refine(n_jobs=n_cpus,verbose=True)
        #p2p_21_icp = model.get_p2p(n_jobs=n_cpus)
        #cmap1 = visu(mesh1.vertlist); cmap2 = cmap1[p2p_21_icp]
        #double_plot(mesh1,mesh2,cmap1,cmap2)
        #return p2p_21_icp, p2p_21
    if refine == 'zoomout':
        model.change_FM_type('classic') # We refine the first computed map, not the icp-refined one
        model.zoomout_refine(nit=11, step = 1,n_jobs=n_cpus,verbose=True)
        #p2p_21_zo = model.get_p2p(n_jobs=n_cpus)
        #cmap1 = visu(mesh1.vertlist); cmap2 = cmap1[p2p_21_zo]
        #double_plot(mesh1,mesh2,cmap1,cmap2)
        #return p2p_21_zo, p2p_21

    print(' Calculating shape distance matrix')
    model.compute_SD()
    #D_area = model.D_a
    #D_conformal = model.D_c
    #print(D_area)
    #print(D_conformal)
    #sns.heatmap(D_conformal)
    return model.D_a, model.D_c, p2p_21

def compute_shape_difference(model):
    """
    Save the mesh as a .off file using a divergent colormap of wks siganute
    
    filename : path with filename to save data
    vertlist : list of mesh vertices
    facelist : list of mesh faces
    wks : wks descriptors 

    """
    model.compute_SD()
    D_area = model.D_a
    D_conformal = model.D_c

    return D_area, D_conformal 
