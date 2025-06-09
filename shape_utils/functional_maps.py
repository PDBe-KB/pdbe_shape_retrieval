import numpy as np
import logging

logger = logging.getLogger()

def visu(vertices):
    
    min_coord,max_coord = np.min(vertices,axis=0,keepdims=True),np.max(vertices,axis=0,keepdims=True)
    cmap = (vertices-min_coord)/(max_coord-min_coord)
    return cmap

def calculate_functional_maps(model,n_cpus = 1, refine= None):
    """                                                                                                                                                                                
    Calculate functional maps and point to point maps with pyFM code (https://github.com/RobinMagnet/pyFM)                                                                                                                                        
                                                                                                                                                                                       
    Returns functional maps and fitted model   

    Args:               
        mesh1 (list) : array with vertices and faces 
        mesh2 (list) : second array with vertices and faces 
        model (int) : functional maps model pyFM
        refine (str) : Selected method to refine functional map                                                                                                                                                           
    Returns:
        FM : Functional map (correspondance matrix)
        p2p21 : Point to point map 
    """   
    print('cpus used', n_cpus)
    fit_params = {
        'w_descr': 1e0,
        'w_lap': 1e-2,
        'w_dcomm': 1e-1,
        'w_orient': 0
    }

    logging.info(f"Computing correspondance matrix ")
    model.fit(**fit_params, verbose=True)

    if refine is None :
        logging.info(f"Computing point to point map using correspondance matrix")
        p2p_21 = model.get_p2p(n_jobs=n_cpus)
        
    #refine model using ICP or Zoom
    if refine == 'icp':
        model.change_FM_type('classic')
        model.icp_refine(n_jobs=n_cpus,verbose=True)
        p2p_21 = model.get_p2p(n_jobs=n_cpus)
        
    if refine == 'zoomout':
        model.change_FM_type('classic') # We refine the first computed map, not the icp-refined one
        model.zoomout_refine(nit=11, step = 1,n_jobs=n_cpus,verbose=True)
        p2p_21 = model.get_p2p(n_jobs=n_cpus)
        #model.compute_SD()
    
    return p2p_21, model.FM


def compute_shape_difference(model):
    """
    Computes shape difference operators, area-based and conformal 
    
    Args:
        model : functional map fitting model computed with pyFM

    Returns:
        D_area : Area-based shape difference operator
        D_conformal : Conformal shape difference operator
    """
    model.compute_SD()
    D_area = model.D_a
    D_conformal = model.D_c

    return D_area, D_conformal 

def calculate_functional_maps_chem(model,descr1,descr2,n_cpus = 1, refine= None):
    """                                                                                                                                                                                
    Calculate functional maps and point to point maps with pyFM code (https://github.com/RobinMagnet/pyFM)                                                                                                                                        
                                                                                                                                                                                       
    Returns functional maps and fitted model   

    Args:               
        mesh1 (list) : array with vertices and faces 
        mesh2 (list) : second array with vertices and faces 
        model (int) : functional maps model pyFM
        refine (str) : Selected method to refine functional map                                                                                                                                                           
    Returns:
        FM : Functional map (correspondance matrix)
        p2p21 : Point to point map 
    """   
    print('cpus used', n_cpus)
    fit_params = {
        'w_descr': 1e0,
        'w_lap': 1e-2,
        'w_dcomm': 1e-1,
        'w_orient': 0
    }
    model.fit_othdescr(descr1,descr2,**fit_params, verbose=True)

    return model.FM
