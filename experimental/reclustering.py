# New file to handle reclustering

from config import params
from clustering_pools import ClusteringMemory
from clustering_mechs import Cluster, ClusteringMechanism
import torch
from torch_kmeans import KMeans


def recluster_memory(memory: ClusteringMemory):
    for pool_label in memory.pools:
        pool = memory.pools[pool_label]
        recluster_pool(pool)


def recluster_pool(pool: ClusteringMechanism):
    samples, labels = pool.get_clusters_with_labels()

    # No samples here!
    if samples.numel() == 0:
        return

    print(f"Pool has {samples.shape[0]} samples with {len(torch.unique(samples, dim=0))} unique samples")
    print(f"Sample variance: {samples.var(dim=0).mean()}")
    print(f"Distance between samples: min={torch.pdist(samples).min()}, max={torch.pdist(samples).max()}")

    # Add batch dimension for KMeans
    # All samples should have the same label because we're clustering by pool
    samples_batched = samples.unsqueeze(0)

    n_clusters = min(pool.Q, samples.shape[0])

    print(f"Input shape to K-means: {samples_batched.shape}, requesting {n_clusters} clusters")

    new_model = KMeans(n_clusters=n_clusters, verbose=True)
    cluster_assignments = new_model.fit_predict(samples_batched)

    # Remove batch dimension
    cluster_assignments = cluster_assignments.squeeze(0)

    print(f"Cluster assignments: {cluster_assignments}")
    print(f"Assignment counts: {torch.bincount(cluster_assignments)}")

    # Check if all samples got assigned to the same cluster
    unique_assignments = torch.unique(cluster_assignments)
    print(f"Number of clusters actually used: {len(unique_assignments)}")





    # Clear ts
    pool.clusters = []

    # Build new clusters directly from K-means output
    for cluster_id in range(n_clusters):
        mask = cluster_assignments == cluster_id
        cluster_samples = samples[mask]
        cluster_labels = [labels[i] for i in range(len(labels)) if mask[i]]

        if len(cluster_samples) > 0:
            # Create new cluster with first sample
            new_cluster = Cluster(cluster_samples[0], cluster_labels[0])

            # Add remaining samples, respecting the P limit
            for i in range(1, len(cluster_samples)):
                new_cluster.add_sample(cluster_samples[i], cluster_labels[i])

                # If cluster exceeds P, remove oldest (same logic as ClusteringMechanism)
                if len(new_cluster.samples) > pool.P:
                    new_cluster.remove_one()

            pool.clusters.append(new_cluster)


# How many samples you need to wait before we can recluster
# Configure in ... config.py. Pretty self explanatory.
def calculate_recluster_interval(freq: int):
    if params['dataset_name'] != ('mnist' or 'fashion_mnist'):
        print("IDK what dataset for the frequency calculation in reclustering.py.")
        print("We're just gonna recluster every 1001 points, if that's cool with you.")
        return 1001
    num_batches = 120000
    x = (num_batches // (freq+1)) + 1 # So that we don't recluster at the end
    return x


class Counter:
    def __init__(self):
        self.count = 0

    def get_count(self):
        return self.count

    def iterate(self):
        self.count += 1

    def reset(self):
        self.count = 0
