import numpy as np
import matplotlib.pyplot as plt
from shape_utils.pyFM_pdbe.mesh import TriMesh
from shape_utils.pyFM_pdbe import functional 
from shape_utils.pyhks import trimesh, hks
from shape_utils.pyhks.trimesh import save_off
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
    enlist = model.energylist

    return descr_1, descr_2, enlist
    
    #    save_to_csv(coords,WKS_1, output_dir)
        
    #    return WKS_1

def saveWKSColors(filename, vertlist, wks, facelist, cmap = 'tab20c'):
    """
    Save the mesh as a .off file using a divergent colormap of wks siganute
    
    filename : path with filename to save data
    vertlist : list of mesh vertices
    facelist : list of mesh faces
    wks : wks descriptors 

    """
    c = plt.get_cmap(cmap)
    x = (wks - np.min(wks))
    x /= np.max(x)
    np.array(np.round(x*255.0), dtype=np.int32)
    C = c(x)
    C = C[:, 0:3]
    save_off(filename, vertlist, C, facelist)

def distance_WKS(wks1,wks2,output_dir):
    """
    Compute distance between two descriptors maps
    Returns file with distance  

    wks1 : list of wks descriptors for mesh1
    wks2 : list of wks descriptors for mesh2

    """
    dist_coefs=[]
    distance_wks = []
    for i,j in zip(wks1,wks2):
        dist_coefs=[]
        coef_sum = 0.0
        for wks_e1,wks_e2 in zip(i,j):
            if abs((wks_e1-wks_e2)/(wks_e1+wks_e2)) > 1.0:
                print('not equal',wks_e1, wks_e2)
            coef = abs(wks_e1-wks_e2)
            coef_sum += coef

            dist_coefs.append(coef)
        d_wks = np.trapz(dist_coefs)
        distance_wks.append(d_wks)
    output_file = os.path.join(output_dir,"dist_wks_maps.dat")
    #save_to_csv(distance_wks, output_file)
    return distance_wks
    
#def calculate_HKS(mesh1,mesh2,kprocess,step, entry1_id,entry2_id,output_path):
#    (VPos1, VColors1, ITris1) = trimesh.load_off(mesh1)
#    (VPos2, VColors2, ITris2) = trimesh.load_off(mesh2)
#    neigvecs1 = min(VPos1.shape[0], kprocess)
#    neigvecs2 = min(VPos2.shape[0], kprocess)
#    descr1 = hks.get_hks(VPos1, ITris1, neigvecs1, np.array([step]))
#    descr2 = hks.get_hks(VPos2, ITris2, neigvecs2, np.array([step]))
#
#    output1=os.path.join(output_path,"{}_hks.dat".format(entry1_id))
#    output2=os.path.join(output_path,"{}_hks.dat".format(entry2_id))
#
#    hks.saveHKSColors(output1, VPos1, descr1[:, 0], ITris1)
#    hks.saveHKSColors(output2, VPos2, descr2[:, 0], ITris2)
