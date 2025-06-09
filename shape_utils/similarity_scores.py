import numpy as np 
from numpy.linalg import inv


def calculate_geodesic_norm_score(FM):
    """                                                                                                                                                                                
    Calculates norm of correspondance matrix based on geodesic distance of eigenvalues spectrum                                                                                                                                
                                                                                                                                                                                       
    Returns norm as a similarity score  

    Args:               
        FM : Correspondance matrix of functional map 
    Returns:
    """  
    
    eigenvalues_FM = np.linalg.eigvals(FM)
    score = np.sqrt(np.sum(np.log(np.absolute(np.real(eigenvalues_FM))) ** 2))

    return score 

def calculate_shape_diff_operator_distance(shape_diffs_ops):
    """                                                                                                                                                                                
    Calculates score based on geodesic distance between shape difference operators (area or comformal)                                                                                                                             
                                                                                                                                                                                       
    Returns geodesic distance as a similarity score  

    Args:               
        shape_diffs_ops : List of shape difference operators with the same base structure
    """      
    invD_set = []
    DinvD_set = []
    eigenvalues_DinvD = []
    geo_distances = []

    for i in range(len(shape_diffs_ops)):

        invD_area = inv(shape_diffs_ops[i])
        invD_set.append(invD_area)

    pairs = get_pairs_two(shape_diffs_ops,invD_set)

    dist_mat_dim = (len(shape_diffs_ops))

    for pair in pairs:
        DinvD = np.matmul(pair[0],pair[1])
        DinvD_set.append(DinvD)

    for elem in DinvD_set:

        eigenvalues = np.linalg.eigvals(elem)
        eigenvalues_DinvD.append(np.real(eigenvalues))
    for eigs in eigenvalues_DinvD:
        result = np.sqrt(np.sum(np.log(np.absolute(np.real(eigs))) ** 2))
        geo_distances.append(result)

    group_size = dist_mat_dim
    dist_matrix = [geo_distances[i:i+group_size] for i in range(0, len(geo_distances), group_size)]
    
    return dist_matrix 