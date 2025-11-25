import numpy as np
import pandas as pd

from scipy.linalg import norm
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn import cluster
from scipy.cluster.hierarchy import dendrogram, from_mlab_linkage, cut_tree, leaves_list, set_link_color_palette, to_tree, fcluster
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import silhouette_samples
import scipy.stats as st
import itertools
from itertools import combinations_with_replacement

def get_pairs_fast(arr):
    return list(combinations_with_replacement(arr, 2))

def get_pairs(arr):
    pairs = []
    for i in range(len(arr)):
        for j in range(i, len(arr)):
            if i <= j:
                pairs.append((arr[i], arr[j]))
    return pairs
def compute_scores_sym_matrix(scores_file, entries_file):
    scores_file=open(scores_file)
    scores_entries = scores_file.read().splitlines()
    entries_file = open(entries_file).read().splitlines()
    entry_labels = [s.split(' ')[0] for s in entries_file]
    
    #Derive dimension of the score matrix from the list of points
    dim = len(entry_labels)
    #Read list of elements and compute pairs 
    pairs_entries = get_pairs_fast(entry_labels)
    #print(entry_labels)
    #print(len(pairs_entries))
    axes_labels = []
    for label in entry_labels:
        axes_labels.append(label)

    scores = []

    for j in pairs_entries:
        j_inv = (j[1],j[0])
        for line in scores_entries:
            p = line.split()
            pair_score = (p[0],p[1])
            score = p[2] 
            if j==pair_score or j_inv==pair_score:
                scores.append(score)
    sym_matrix = np.zeros((dim, dim))
    row, col = np.triu_indices(dim)  # Upper triangular indices
    sym_matrix[row, col] = scores
    sym_matrix[col, row] = scores
   
    return sym_matrix, axes_labels

def compute_scores_sym_matrix_fast(scores_file, entries_file):
    
    entries_file = open(entries_file).read().splitlines()
    entry_labels = [s.split(' ')[0] for s in entries_file]

    dim = len(entry_labels)
    label_to_idx = {label: idx for idx, label in enumerate(entry_labels)}

    # Initialize the symmetric matrix
    sym_matrix = np.zeros((dim, dim))

    # Build a dictionary for fast lookup
    scores_dict = {}
    with open(scores_file) as f:
        for line in f:
            p1, p2, score = line.strip().split()
            score = float(score)
            scores_dict[(p1, p2)] = score
            scores_dict[(p2, p1)] = score  # Enforce symmetry in lookup

    # Get pairs (upper triangle including diagonal)
    pairs_entries = get_pairs(entry_labels)

    # Fill the matrix using pairs
    for (p1, p2) in pairs_entries:
        if (p1, p2) in scores_dict:
            i = label_to_idx[p1]
            j = label_to_idx[p2]
            sym_matrix[i, j] = scores_dict[(p1, p2)]
            sym_matrix[j, i] = scores_dict[(p1, p2)]  # Enforce symmetry
        else:
            # Optional: Handle missing pairs if needed (e.g., assign zero or NaN)
            pass

    return sym_matrix, entry_labels    

def compute_partial_scores_matrix_fast(scores_file, entries_file, fill_value=0.0):
    # Read entry labels (subset allowed)
    with open(entries_file) as f:
        entry_labels = [line.split()[0] for line in f]
    
    dim = len(entry_labels)
    label_to_idx = {label: idx for idx, label in enumerate(entry_labels)}

    sym_matrix = np.full((dim, dim), fill_value, dtype=float)

    # Build score dictionary, but skip scores not in subset
    scores_dict = {}
    with open(scores_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            p1, p2, score = parts
            if p1 not in label_to_idx or p2 not in label_to_idx:
                continue  # skip scores involving labels outside subset
            score = float(score)
            scores_dict[(p1, p2)] = score
            scores_dict[(p2, p1)] = score

    # Fill matrix
    for p1, p2 in itertools.combinations_with_replacement(entry_labels, 2):
        if (p1, p2) in scores_dict:
            score = scores_dict[(p1, p2)]
            i, j = label_to_idx[p1], label_to_idx[p2]
            sym_matrix[i, j] = score
            sym_matrix[j, i] = score

    return sym_matrix, entry_labels

def linkage_matrix(model):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    return linkage_matrix
    # Plot the corresponding dendrogram
    #dendrogram(linkage_matrix, **kwargs)

def two_gap_diff_stat(model, max_k,dist):
    clusters = linkage_matrix(model)
    dist = pd.DataFrame(dist)
    # cluster levels over from 1 to N-1 clusters
    cluster_lvls = pd.DataFrame(cut_tree(clusters))
    num_k = cluster_lvls.columns  # save column with number of clusters
    # reverse order to start with 1 cluster
    cluster_lvls = cluster_lvls.iloc[:, ::-1]
    cluster_lvls.columns = num_k  # set columns to number of cluster
    W_list = []

    # get within-cluster dissimilarity for each k
    for k in range(min(len(cluster_lvls.columns), max_k)):
        level = cluster_lvls.iloc[:, k]  # get k clusters
        D_list = []  # within-cluster distance list

        for i in range(np.max(level.unique()) + 1):
            cluster = level.loc[level == i]
            # Based on correlation distance
            cluster_dist = dist.loc[cluster.index,
                                    cluster.index]  # get distance
            cluster_pdist = squareform(cluster_dist, checks=False)
            if cluster_pdist.shape[0] != 0:
                D = np.nan_to_num(cluster_pdist.mean())
                D_list.append(D)  # append to list

        W_k = np.sum(D_list)
        W_list.append(W_k)

    W_list = pd.Series(W_list)
    n = dist.shape[0]
    limit_k = int(min(max_k, np.sqrt(n)))
    gaps = W_list.shift(2) + W_list - 2 * W_list.shift(1)
    gaps = gaps[0:limit_k]
    if gaps.isna().all():
        k = len(gaps)
    else:
        k = int(gaps.idxmax() + 2)

    return k


def std_silhouette_score(model, max_k,dist):
    clusters = linkage_matrix(model)
    dist = pd.DataFrame(dist)
    # cluster levels over from 1 to N-1 clusters
    cluster_lvls = pd.DataFrame(cut_tree(clusters))
    num_k = cluster_lvls.columns  # save column with number of clusters
    # reverse order to start with 1 cluster
    cluster_lvls = cluster_lvls.iloc[:, ::-1]
    cluster_lvls.columns = num_k  # set columns to number of cluster
    scores_list = []

    # get within-cluster dissimilarity for each k
    for k in range(2, min(len(cluster_lvls.columns), max_k)):
        level = cluster_lvls.iloc[:, k]  # get k clusters
        b = silhouette_samples(dist, level)
        scores_list.append(b.mean() / b.std())

    scores_list = pd.Series(scores_list)
    n = dist.shape[0]
    limit_k = int(min(max_k, np.sqrt(n)))
    scores_list = scores_list[0:limit_k]
    if scores_list.isna().all():
        k = len(scores_list)
    else:
        k = int(scores_list.idxmax() + 2)

    return k


def find_optimal_num_clusters(model, dist, max_k=10, ktype="d",k=None):
    if k is None:
        if ktype == "s":
            k = std_silhouette_score(model, max_k,dist)
        else:
            k = two_gap_diff_stat(model, max_k,dist)

    return k

def compute_clusters(sym_matrix,axes_labels,cluster,linkage_method="ward", threshold=None,no_clusters=None):
    if linkage_method == "average":
        average_linkage_average = cluster.AgglomerativeClustering(
        linkage=linkage_method,
        metric = 'precomputed',
        compute_distances = True,
        compute_full_tree = True, 
        distance_threshold = threshold,
        n_clusters= no_clusters,
    )
    if linkage_method == "ward":
        average_linkage_average = cluster.AgglomerativeClustering(
            linkage='ward',
            compute_distances = True,
            compute_full_tree = True, 
            distance_threshold = None,
            n_clusters= no_clusters,
        )
    

    clustering_av = average_linkage_average.fit(sym_matrix)
    k = clustering_av.n_clusters_
    threshold_dist = average_linkage_average.distances_[-(k-1)]
    
    k = clustering_av.n_clusters_

    #print('no. of clusters',clusters)
    link_matrix = linkage_matrix(clustering_av)
    clustering_inds = fcluster(link_matrix, k, criterion="maxclust")
    clusters = {i: [] for i in range(min(clustering_inds), max(clustering_inds) + 1)}
    for i, v in enumerate(clustering_inds):
        clusters[v].append(i)

    clusters_all = []
    for i in range(1,len(clusters)+1):
        cluster_entries = []
        cluster = clusters[i]
        for j in cluster:
            #print(axes_labels[j])
            cluster_entries.append(axes_labels[j])  
        clusters_all.append(cluster_entries)
    
    return clusters_all,k,link_matrix,threshold_dist

def compute_volumes_pockets(volumes_file, entries_cluster):
    volumes_file = open(volumes_file)
    volumes_pockets = volumes_file.read().splitlines()
    entry_labels = entries_cluster
    
    #Derive dimension of the score matrix from the list of points
    dim = len(entry_labels)
    
    axes_labels = []
    for label in entry_labels:
        axes_labels.append(label)

    vol_pockets = []

    for j in entry_labels:
        for line in volumes_pockets:
            p = line.split()
            pocket_entry = p[0]
            pocket_volume = p[1] 
            if j==pocket_entry :
                vol_pockets.append([pocket_entry,pocket_volume])   

    return vol_pockets