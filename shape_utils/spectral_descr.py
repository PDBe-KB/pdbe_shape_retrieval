import numpy as np
import logging
import os 


logger = logging.getLogger()

def calculate_descriptors(model,kprocess,n_ev,ndescr,step,landmarks, output_dir,descr_type='WKS'):
    """                                                                                                                                                                                
    Calculate Wave Kernel Signatures and Heat Kernel Signatures on triangulated meshes using pyFM (https://github.com/RobinMagnet/pyFM)                                                                                                                                        
                                                                                                                                                                                       
    Returns list of WKS descriptors for each mesh    

    Args:  
        model : FunctionalMapping and Trimesh model calculated with pyFM module             
        kprocess (int) : No. of eigenvalues to use      
        n_ev : the least number of Laplacian eigenvalues to consider                                                                                                                                                    
        ndescr (int) : No. of descriptors to include
        step (int)   : sub-sample step, in order to not use too many descriptors
        landmarks  : (p,1|2) array of indices of landmarks to match.
                        If (p,1) uses the same indices for both.
        output_dir : path to output directory
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

    #preprocess functional mapping and compute descriptors
    
    model.preprocess(**process_params,verbose=True)

    #wks descriptors for surface meshes (mesh1 and mesh2)

    descr_1 = model.descr1
    descr_2 = model.descr2
    #enlist = model.energylist

    return descr_1, descr_2


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
                    print('not equal',wks_e1, wks_e2)
            coef = abs(wks_e1-wks_e2)
            coef_sum += coef
        distance_wks.append(coef_sum)
    return distance_wks

