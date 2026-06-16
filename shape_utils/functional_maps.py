from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pyFM import functional
import dense_mesh as dm
import pyFM.spectral as spectral
from pyFM.spectral.nn_utils import knn_query

from shape_utils.config import DenseMeshConfig, FunctionalMapConfig


logger = logging.getLogger(__name__)

def set_FM_model_parameters(
    mesh1: Any,
    mesh2: Any,
    kprocess: int,
    n_ev: int,
    ndescr: int,
    step: int,
    landmarks: Any,
    descr_type: str = 'WKS',
) -> Any:
    """                                                                                                                                                                                
    Set model parameters for FM and Wave Kernel Signatures and Heat Kernel Signatures on triangulated meshes using pyFM (https://github.com/RobinMagnet/pyFM)                                                                                                                                        
                                                                                                                                                                                       
    Returns FM model with list of WKS descriptors for each mesh  

    Args:  
        mesh1 (list) : array with vertices and faces
        mesh2 (list) : second array with vertices and faces
        kprocess (int) : No. of eigenvalues to use      
        n_ev : the least number of Laplacian eigenvalues to consider                                                                                                                                                    
        ndescr (int) : No. of descriptors to include
        step (int)   : sub-sample step, in order to not use too many descriptors
        landmarks  : (p,1|2) array of indices of landmarks to match.
                        If (p,1) uses the same indices for both.
        descr_type : Descriptor type : WKS (default), HKS or Zernike 
    outputs:
        model : functional map model with set parameters and computed descriptors
    """   
    
    process_params = {
        'n_ev': (n_ev,n_ev), # n_ev: (k1, k2) tuple - with the least number of Laplacian eigenvalues to consider.
        'subsample_step': int(step),  # In order not to use too many descriptors
        'descr_type': descr_type,  # WKS or HKS
        'k_process' : int(kprocess),    # No. of eigenvalues/eigenvectors to compute 
        'n_descr': int(ndescr),        #
        'landmarks': landmarks
    }

    model = functional.FunctionalMapping(mesh1,mesh2) 

    #preprocess functional mapping and compute descriptors
    
    model.preprocess(**process_params,verbose=True)
    
    #enlist = model.energylist

    return model

def calculate_functional_maps(
    model: Any,
    config: FunctionalMapConfig | dict[str, Any] | int | None = None,
    *,
    n_cpus: int | None = None,
    refine: str | None = None,
) -> tuple[Any, np.ndarray]:
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
    fm_config = FunctionalMapConfig.from_value(config, n_cpus=n_cpus, refine=refine)
    logger.info("cpus used: %s", fm_config.n_cpus)

    logger.info("Computing correspondence matrix")
    model.fit(**fm_config.fit_params(), verbose=fm_config.verbose)

    if fm_config.refine is None:
        logger.info("Computing point to point map using correspondence matrix")
        FM = model.FM
    elif fm_config.refine == 'icp':
        model.change_FM_type('classic')
        model.icp_refine(n_jobs=fm_config.n_cpus, verbose=fm_config.verbose)
        FM = model.FM
    elif fm_config.refine == 'zoomout':
        model.change_FM_type('classic') # We refine the first computed map, not the icp-refined one
        model.zoomout_refine(
            nit=fm_config.zoomout_nit,
            step=fm_config.zoomout_step,
            verbose=fm_config.verbose,
        )
        FM = model.FM
    else:
        raise ValueError("refine must be one of: icp, zoomout, or None")
        
    model_FM = model

    return model_FM, FM

def calculate_p2p_map(model_FM: Any, n_cpus: int = 8) -> np.ndarray:
    p2p_21 = model_FM.get_p2p(n_jobs=n_cpus)
    return p2p_21

def calculate_scalable_functional_maps(
    mesh1: Any,
    mesh2: Any,
    neigvecs: int,
    n_samples: int,
    n_cpus: int = 8,
    dense_mesh_config: DenseMeshConfig | None = None,
) -> np.ndarray:
    process_params = (dense_mesh_config or DenseMeshConfig()).process_params(n_cpus)


    U1, Ab1, Wb1, sub1, distmat1 = dm.process_mesh(mesh1, n_samples, **process_params)
    evals1, evects1 = dm.get_approx_spectrum(Wb1, Ab1, k=neigvecs, verbose=True)

    U2, Ab2, Wb2, sub2, distmat2 = dm.process_mesh(mesh2, n_samples, **process_params)
    evals2, evects2 = dm.get_approx_spectrum(Wb2, Ab2, k=neigvecs, verbose=True)


    # Compute an initial approximate functional map
    p2p_21_sub_init = knn_query(mesh1.vertices[sub1], mesh2.vertices[sub2], k=1, n_jobs=n_cpus)
    # We compute the initial functional map using the approximate spectrum here (same method that will be used inside ZoomOut)
    FM_12_init = spectral.p2p_to_FM(
    p2p_21_sub_init, evects1[:, :20], evects2[:, :20], A2=Ab2
    )

    #FM_12_zo, p2p_21_sub_zo = zoomout_refine(
    #FM_12_init,
    #evects1,
    #evects2,
    #nit=16,
    #step=5,
    #A2=Ab2,
    #return_p2p=True,
    #n_jobs=n_cpus,
    #verbose=True,
    #)

    return FM_12_init


def compute_shape_difference(model: Any) -> tuple[np.ndarray, np.ndarray]:
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

def calculate_functional_maps_chem(
    model: Any,
    descr1: np.ndarray,
    descr2: np.ndarray,
    config: FunctionalMapConfig | dict[str, Any] | int | None = None,
    *,
    n_cpus: int | None = None,
    refine: str | None = None,
) -> np.ndarray:
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

    fm_config = FunctionalMapConfig.from_value(config, n_cpus=n_cpus, refine=refine)
    logger.info("cpus used: %s", fm_config.n_cpus)
    model.fit_othdescr(descr1, descr2, **fm_config.fit_params(), verbose=fm_config.verbose)

    return model.FM
