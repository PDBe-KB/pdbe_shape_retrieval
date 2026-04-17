import numpy as np
import logging
import os 
from pyFM import functional


logger = logging.getLogger()

def calculate_spectral_descriptors(mesh,kprocess,n_ev,ndescr,step,landmarks,descr_type='WKS'):
    """                                                                                                                                                                                
    Calculate Wave Kernel Signatures and Heat Kernel Signatures on triangulated meshes using pyFM (https://github.com/RobinMagnet/pyFM)                                                                                                                                        
                                                                                                                                                                                       
    Returns list of WKS descriptors for each mesh    

    Args:  
        mesh : Trimesh mesh object
        kprocess (int) : No. of eigenvalues to use      
        n_ev : the least number of Laplacian eigenvalues to consider                                                                                                                                                    
        ndescr (int) : No. of descriptors to include
        step (int)   : sub-sample step, in order to not use too many descriptors
        landmarks  : (p,1|2) array of indices of landmarks to match.
                     If (p,1) uses the same indices for both.
        descr_type : Descriptor type : WKS (default), HKS or Zernike 
 
    """   
    
    process_params = {
        'n_ev': (n_ev,n_ev), # n_ev: (k1, k2) tuple - with the least number of Laplacian eigenvalues to consider.
        'subsample_step': int(step),  # In order not to use too many descriptors
        'descr_type': descr_type,  # WKS or HKS
        'k_process' : int(kprocess),    # No. of eigenvalues/eigenvectors to compute 
        'n_descr': int(ndescr),        #
        'landmarks': landmarks
    }

    descr_model = functional.SpectralDescriptors(mesh) 

    #preprocess functional mapping and compute descriptors
    descr_model.preprocess_descriptors_mesh(**process_params,verbose=True)
    
    #wks descriptors for surface mesh (mesh1 and mesh)

    descr = descr_model.descr
     
    return descr 


def distance_WKS(wks1,wks2):
    """
    Compute distance between two descriptors maps
    Returns file with distance  

    wks1 : list of wks descriptors for mesh1
    wks2 : list of wks descriptors for mesh2

    """
    distance_wks = []
    for i,j in zip(wks1,wks2):
        coef_sum = 0.0
        for wks_e1,wks_e2 in zip(i,j):
            denom = wks_e1 + wks_e2
            if denom != 0:
                if abs((wks_e1-wks_e2)/(wks_e1+wks_e2)) > 1.0:
                    logging.info('not equal {} and {}'.format(wks_e1, wks_e2))
            coef = abs(wks_e1-wks_e2)
            coef_sum += coef
        distance_wks.append(coef_sum)
    return distance_wks

