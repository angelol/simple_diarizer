import numpy as np

import scipy.cluster.hierarchy as hcluster
from scipy.sparse.csgraph import laplacian
from scipy.ndimage import gaussian_filter
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def similarity_matrix(embeds, metric="cosine"):
    return pairwise_distances(embeds, metric=metric)


def cluster_AHC(embeds, n_clusters=None, threshold=None, metric="cosine", **kwargs):
    """
    Cluster embeds using Agglomerative Hierarchical Clustering
    """
    if n_clusters is None:
        assert threshold, "If num_clusters is not defined, threshold must be defined"

    S = similarity_matrix(embeds, metric=metric)

    if n_clusters is None:
        cluster_model = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="average",
            compute_full_tree=True,
            distance_threshold=threshold,
        )

        return cluster_model.fit_predict(S)
    else:
        cluster_model = AgglomerativeClustering(
            n_clusters=n_clusters, affinity="precomputed", linkage="average"
        )

        return cluster_model.fit_predict(S)


##########################################
# Spectral clustering
# A lot of these methods are lifted from
# https://github.com/wq2012/SpectralCluster
##########################################

def compute_n_clusters(embeds, threshold, period):
    """
    Compute the number of clusters
    """
    # five_minutes = 300
    # observation_frames = int(five_minutes / period)
    torch_embeds = torch.from_numpy(embeds).to(device)
    S = compute_affinity_matrix(torch_embeds)
    S = sim_enhancement(S)
    eigenvalues = compute_sorted_eigenvalues(S)
    n_clusters = compute_number_of_clusters(eigenvalues, 100, threshold)
    return n_clusters


def cluster_SC(embeds, n_clusters=None, threshold=1e-2, enhance_sim=True, period=0.5, **kwargs):
    """
    Cluster embeds using Spectral Clustering
    """
    if n_clusters is None:
        n_clusters = compute_n_clusters(embeds, threshold, period)

    print('calling SpectralClustering n_clusters found: ', n_clusters)
    cluster_model = SpectralClustering(
        n_clusters=n_clusters, affinity="nearest_neighbors",
        eigen_solver="lobpcg", assign_labels='cluster_qr'
    )
    if n_clusters == 1:
        return np.zeros(len(embeds))
    else:
        return cluster_model.fit_predict(embeds)


def diagonal_fill(A):
    """
    Sets the diagonal elements of the matrix to the max of each row
    """
    diag_idx = torch.arange(0, A.shape[0], out=torch.LongTensor()).to(A.device)
    A[diag_idx, diag_idx] = 0.0
    A[diag_idx, diag_idx] = torch.max(A, dim=1).values
    return A


def gaussian_blur(A, sigma=1):
    """
    Does a Gaussian blur on the affinity matrix
    """
    return torchvision.transforms.functional.gaussian_blur(A.unsqueeze(0), (sigma, sigma))[0]


def row_threshold_mult(A, p=0.95, mult=0.01):
    """
    For each row multiply elements smaller than the row's p'th percentile by mult
    """
    percentiles = torch.quantile(A, p, dim=1, keepdim=True)
    mask = A < percentiles
    A = (mask * mult * A) + (~mask * A)
    return A


def symmetrization(A):
    """
    Symmeterization: Y_{i,j} = max(S_{ij}, S_{ji})
    """
    return torch.max(A, A.t())


def diffusion(A):
    """
    Diffusion: Y <- YY^T
    """
    return torch.matmul(A, A.t())


def row_max_norm(A):
    """
    Row-wise max normalization: S_{ij} = Y_{ij} / max_k(Y_{ik})
    """
    maxes = torch.max(A, dim=1, keepdim=True).values
    return A / maxes


def sim_enhancement(A):
    func_order = [
        diagonal_fill,
        gaussian_blur,
        row_threshold_mult,
        symmetrization,
        diffusion,
        row_max_norm,
    ]
    for f in func_order:
        A = f(A)
    return A


def compute_affinity_matrix(X):
    """Compute the affinity matrix from data.
    Note that the range of affinity is [0,1].
    Args:
    X: PyTorch tensor of shape (n_samples, n_features)
    Returns:
    affinity: PyTorch tensor of shape (n_samples, n_samples)
    """
    # Normalize the data.
    l2_norms = torch.norm(X, dim=1)
    X_normalized = X / l2_norms[:, None]
    # Compute cosine similarities. Range is [-1,1].
    cosine_similarities = torch.mm(X_normalized, X_normalized.T)
    # cosine_similarities = (cosine_similarities + 1.0) / 2.0
    return cosine_similarities


def compute_sorted_eigenvalues(A):
    """Sort eigenvectors by the real part of eigenvalues.
    Args:
        A: the matrix to perform eigen analysis with shape (M, M)
    Returns:
        w: sorted eigenvalues of shape (M,)
        v: sorted eigenvectors, where v[;, i] corresponds to ith largest
           eigenvalue
    """

    # Eigen decomposition.
    eigenvalues = torch.linalg.eigvalsh(A)
    # Sort from largest to smallest.
    index_array = torch.argsort(-eigenvalues, dim=0, descending=False)
    # Re-order.
    w = eigenvalues[index_array]
    return w


def compute_number_of_clusters(eigenvalues, max_clusters=None, threshold=1e-2):
    """Compute number of clusters using EigenGap principle.
    Args:
        eigenvalues: sorted eigenvalues of the affinity matrix
        max_clusters: max number of clusters allowed
        stop_eigenvalue: we do not look at eigen values smaller than this
    Returns:
        number of clusters as an integer
    """
    if not threshold:
        threshold = 1e-2
    max_delta = 0
    max_delta_index = 0
    range_end = len(eigenvalues)
    if max_clusters and max_clusters + 1 < range_end:
        range_end = max_clusters + 1
    for i in range(1, range_end):
        print('eigenvalues[i - 1]: ', eigenvalues[i - 1],
              ' threshold: ', threshold)
        if eigenvalues[i - 1] < threshold:
            break
        delta = eigenvalues[i - 1] / eigenvalues[i]
        print(
            f'i: {i} delta: {delta} max_delta: {max_delta} max_delta_index: {max_delta_index}')
        if delta > max_delta:
            max_delta = delta
            max_delta_index = i
    return max_delta_index
