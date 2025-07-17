import numpy as np 
from numpy.linalg import inv
from torch import FloatTensor, LongTensor
from shape_utils.models import SimpleEuclideanModel, NeuralNetworkModel
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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


def get_pairs(arr):
    pairs = []
    for i in range(len(arr)):
        for j in range(i, len(arr)):
            if i <= j:
                pairs.append((arr[i], arr[j]))
    return pairs

def read_inv(fn):
    vectors = []
    f = open(fn, 'r')
    for line in f:
        vectors.append(float(line.strip()))
    f.close()
    return vectors[1::]
def read_ply(fn):
    element_vertex = None
    element_face = None
    f = open(fn, 'r')
    for line in f:
        line = line.strip()
        if 'element vertex' in line:
            element_vertex = int(line.split()[-1])
        if 'element face' in line:
            element_face = int(line.split()[-1])
        if element_vertex != None and element_face != None:
            return [element_vertex, element_face]
    f.close()
    return [element_vertex, element_face]

def read_dataset(input_dir,db_structures, atom_type):
    dataset = {}
    for struct in db_structures:
        if atom_type == 'mainchain':
            _3dzd = read_inv(input_dir + struct + '_cacn.inv')
        else:
            _3dzd = read_inv(input_dir + struct + '.inv')

        #_vertex = read_ply(input_dir + struct + '.ply')
        data = {}
        data['_3dzd'] = _3dzd
        #data['vertex_face'] = _vertex

        dataset[struct] = data
    return dataset

def pairs_to_features(pairs,alpha_data,scope_data):

    _3DZD_vectors_1, _3DZD_vectors_2 = [], []
    #element_vertices_1, element_vertices_2 = [], []
    #element_faces_1, element_faces_2 = [], []
    for _pair in pairs:
        id_0, id_1 = str(_pair[0]), str(_pair[1])

        _3dzd_1 = list(alpha_data[id_0]['_3dzd'])
        _3dzd_2 = list(scope_data[id_1]['_3dzd'])


        _3DZD_vector_1 = np.asarray(_3dzd_1)
        _3DZD_vector_1 = np.expand_dims(_3DZD_vector_1.squeeze(), axis = 0)
        _3DZD_vector_2 = np.asarray(_3dzd_2)
        _3DZD_vector_2 = np.expand_dims(_3DZD_vector_2.squeeze(), axis = 0)

        #element_vertex_1, element_face_1 = tuple([int(x) for x in alpha_data[id_0]['vertex_face']])
        #element_vertex_2, element_face_2 = tuple([int(x) for x in scope_data[id_1]['vertex_face']])

        # Update                                                                                     
        _3DZD_vectors_1.append(_3DZD_vector_1)
        _3DZD_vectors_2.append(_3DZD_vector_2)
        #element_vertices_1.append(element_vertex_1)
        #element_faces_1.append(element_face_1)
        #element_vertices_2.append(element_vertex_2)
        #element_faces_2.append(element_face_2)

    _3DZD_vectors_1 = FloatTensor(_3DZD_vectors_1).squeeze()
    #element_vertices_1 = FloatTensor(element_vertices_1)
    #element_faces_1 = FloatTensor(element_faces_1)

    _3DZD_vectors_2 = FloatTensor(_3DZD_vectors_2).squeeze()
    #element_vertices_2 = FloatTensor(element_vertices_2)
    #element_faces_2 = FloatTensor(element_faces_2)

    #vertices_diff = torch.abs(element_vertices_1 - element_vertices_2).unsqueeze(1)
    #faces_diff = torch.abs(element_faces_1 - element_faces_2).unsqueeze(1)
    #extra_features = torch.cat([vertices_diff, faces_diff], dim  = 1)

    return _3DZD_vectors_1, _3DZD_vectors_2

def predict_similarity_zernike(input_dir,output_dir,model_type='simple_euclidean_model', atom_type='fullatom',cuda='true',device_id='0'):

    if cuda == 'true' and torch.cuda.is_available():
        cuda = True
        device_id = device_id
        #device_id = args.device_id                                                                                    
    else:
        cuda = False
        device_id = torch.device("cpu")

    #model_type = 'neural_network'                                                                                     
    atom_type = atom_type
    if model_type == 'neural_network':
        print("neural network of Kihara not yet implemented")
    elif model_type == 'simple_euclidean_model':
        print('Simple Euclidean Model')
        model = SimpleEuclideanModel()
    model.eval()
    db_structures = [x for x in os.listdir(input_dir) if '.inv' in x and '_cacn' not in x]
    db_structures = [x.split('.')[0] for x in db_structures]
    print('pdb to compare : ',len(db_structures))
    if len(db_structures) == 0:
        print('There are no structure to compare.')
        exit()

    database_dataset = read_dataset(input_dir,db_structures,atom_type)
    #query_pdb = db_structures[0]                                                                                      
    my_pairs = get_pairs(db_structures)
    #my_pairs = [(query_pdb, j) for j in db_structures]                                                                
    inputs_1, inputs_2 = pairs_to_features(my_pairs,database_dataset,database_dataset)
    if cuda:
        inputs_1 = inputs_1.cuda(device_id)
        inputs_2 = inputs_2.cuda(device_id)
        #extra_features = extra_features.cuda(device_id)

    if model_type == 'simple_euclidean_model':
        outputs = model(inputs_1, inputs_2, True)
        outputs = outputs.squeeze().cpu().data.numpy().tolist()

    with open(output_dir + atom_type + '_prediction.txt','w') as fh:
        fh.write('Query\tTarget\tDis-similarity Probability\n')
        #for i in range(0,len(outputs)):                                                                               
        for pair,score in zip(my_pairs,outputs):
            fh.write(pair[0] + '\t' + pair[1] + '\t' +str(round(score,3)) + '\n')
            #fh.write(db_structures[0] + '\t' + db_structures[i] + '\t' +str(round(outputs[i],3)) + '\n')              




