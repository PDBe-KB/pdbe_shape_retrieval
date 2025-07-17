import trimesh
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import squareform
import argparse
import logging

from matplotlib import pyplot as plt

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn import cluster
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from scipy.cluster.hierarchy import dendrogram, from_mlab_linkage, cut_tree, leaves_list, set_link_color_palette, to_tree, fcluster
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import silhouette_samples
import scipy.stats as st
import pandas as pd


def load_off_mesh(file_path):
    """Load a .off file and return the mesh."""
    return trimesh.load(file_path, file_type='off')


def get_pairs(arr):
    pairs = []
    for i in range(len(arr)):
        for j in range(i, len(arr)):
            if i <= j:
                pairs.append((arr[i], arr[j]))
    return pairs
def get_sym_square_matrix(dim,pairs_array):
    sym_matrix = np.zeros((dim, dim))
    row, col = np.triu_indices(dim)  # Upper triangular indices
    sym_matrix[row, col] = pairs_array
    sym_matrix[col, row] = pairs_array

    return sym_matrix



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

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-i",
        "--input_pairs_dist",
        help="input file containing the pairs labels and distance/score  ",
        required=True,
    )

    args = parser.parse_args()
    
    pocket_meshes_list = open(str(args.input_list_meshes))

    list_pockets = pocket_meshes_list.read().splitlines()
    list_labels = []

    for entry in list_pockets:
        entry_label = entry[:-4]
        list_labels.append(entry_label)

    dim = len(list_labels)

    pocket_pairs=get_pairs(list_pockets)
    print(len(pocket_pairs))
    min_distance_pockets = []
    for pair in pocket_pairs:
        # Load the meshes
        mesh1 = load_off_mesh(pair[0])
        mesh2 = load_off_mesh(pair[1])   

        # Calculate minimum distance
        min_distance = find_minimum_distance(mesh1, mesh2)
        min_distance_pockets.append(min_distance)
        print(f"The minimum distance between the meshes is: {min_distance}")

    sym_matrix = np.zeros((dim, dim))
    row, col = np.triu_indices(dim)  # Upper triangular indices
    sym_matrix[row, col] = min_distance_pockets
    sym_matrix[col, row] = min_distance_pockets

    print(sym_matrix)
if "__main__" in __name__:
    main()