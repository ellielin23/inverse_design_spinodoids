# utils/visualize_latent.py

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def compute_latent_representations(encoder, P_all, S_all, device="cpu"):
    """
    Computes the mean latent vectors for each (S, P) pair using the encoder.
    Returns a numpy array of shape (num_samples, latent_dim).
    """
    mu_list = []
    with torch.no_grad():
        for i in range(len(P_all)):
            S_i = S_all[i].unsqueeze(0).to(device)
            P_i = P_all[i].unsqueeze(0).to(device)
            mu, _ = encoder(S_i, P_i)
            mu_list.append(mu.cpu().numpy())
    return np.vstack(mu_list)


def reduce_latent_dim(Z_mu, method="tsne", seed=42):
    """
    Projects latent space to 2D using t-SNE or PCA.
    """
    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=seed)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Invalid method. Choose 'tsne' or 'pca'")
    return reducer.fit_transform(Z_mu)


def cluster_latents(Z_mu, n_clusters=3, seed=42):
    """
    Clusters latent space using KMeans. Returns cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    return kmeans.fit_predict(Z_mu)



def plot_latents(Z_2D, cluster_labels=None, cluster_colors=None):
    """
    Plots the 2D latent space. If cluster_labels are provided, colors by cluster
    and prints cluster-color mapping.
    """
    plt.figure(figsize=(6, 5))

    if cluster_labels is None:
        plt.scatter(Z_2D[:, 0], Z_2D[:, 1], alpha=0.6, s=20, color='skyblue')
        plt.title("Latent Space Representation (t-SNE)")
    else:
        if cluster_colors is None:
            cluster_colors = ['deepskyblue', 'lightcoral', 'lightgreen']

        # print cluster-color mapping
        print("Cluster-color mapping:")
        for i, color in enumerate(cluster_colors):
            print(f"    - Cluster {i}: {color}")

        colors = [cluster_colors[label] for label in cluster_labels]
        plt.scatter(Z_2D[:, 0], Z_2D[:, 1], c=colors, s=20, alpha=0.7)
        plt.title("t-SNE Colored by KMeans Cluster")

    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def extract_cluster_region(Z_2D, cluster_labels, target_cluster, x_bounds, y_bounds, cluster_colors=None):
    """
    Extracts indices of points within a specific cluster and bounding box.
    Prints info about selection criteria.
    """
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    region_mask = (
        (Z_2D[:, 0] >= x_min) & (Z_2D[:, 0] <= x_max) &
        (Z_2D[:, 1] >= y_min) & (Z_2D[:, 1] <= y_max) &
        (cluster_labels == target_cluster)
    )

    print("\nExtracting region:")
    print(f"  → Target Cluster: {target_cluster}")
    if cluster_colors is not None:
        try:
            print(f"  → Cluster Color: {cluster_colors[target_cluster]}")
        except IndexError:
            print("  → Cluster Color: [out of range]")
    print(f"  → X bounds: {x_min} to {x_max}")
    print(f"  → Y bounds: {y_min} to {y_max}")

    return np.where(region_mask)[0]


def print_region_examples(region_indices, S_all, P_all, max_examples=5):
    """
    Prints (S, P) pairs from selected region indices.
    """
    print(f"\nShowing up to {max_examples} region examples:")
    for idx in region_indices[:max_examples]:
        print(f"  S[{idx}] = {S_all[idx].numpy()}")
        print(f"  P[{idx}] = {P_all[idx].numpy()}\n")
