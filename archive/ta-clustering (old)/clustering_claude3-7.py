# Initial Code by claude 3.7. 1x1 matrix per item
import numpy as np

class ClusteringMechanism:
    def __init__(self, max_clusters=100, max_cluster_size=3):
        self.clusters = []  # Empty set of clusters
        self.Q = max_clusters  # Maximum number of clusters
        self.P = max_cluster_size  # Maximum cluster size
        self.means = []  # Cluster means
        
    def add(self, z):
        """Add a sample z to the pool with clustering"""
        if len(self.clusters) < self.Q:
            # Initialize a new cluster
            self.clusters.append([z])
            self.means.append(z)
        else:
            # Find closest cluster
            distances = [np.linalg.norm(z - mean, ord=2) for mean in self.means]
            q_star_idx = np.argmin(distances)
            
            # Add z to the closest cluster
            self.clusters[q_star_idx].append(z)
            
            # Remove oldest member if cluster size exceeds maximum
            if len(self.clusters[q_star_idx]) > self.P:
                self.clusters[q_star_idx].pop(0)  # Remove oldest element
            
            # Update the mean of the cluster
            self.means[q_star_idx] = np.mean(self.clusters[q_star_idx], axis=0)

# Example Usage for ClusteringMechanism:
if __name__ == '__main__':
    clustering_system = ClusteringMechanism(max_clusters=5, max_cluster_size=2) # Smaller Q, P for demonstration
    mean = lambda i:sum(i)/len(i) if len(i)>0 else 0

    print("Adding first 5 samples to initialize clusters:")
    for i in range(5):
        sample = np.array([i, i + 0.5])
        clustering_system.add(sample)
        print(f"Added sample {sample}. Clusters: {len(clustering_system.clusters)}")
        for j, c in enumerate(clustering_system.clusters):
            print(f"  Cluster {j} (mean: {mean(c)}): {list(c.samples)}")

    print("\nAdding more samples, testing cluster size limit and mean update:")
    # These samples should go into existing clusters, triggering removal of oldest
    sample_close_to_0 = np.array([0.1, 0.6])
    clustering_system.add(sample_close_to_0)
    print(f"Added sample {sample_close_to_0}. Clusters: {len(clustering_system.clusters)}")
    for j, c in enumerate(clustering_system.clusters):
        print(f"  Cluster {j} (mean: {mean(c)}): {list(c.samples)}")

    sample_close_to_4 = np.array([4.1, 4.6])
    clustering_system.add(sample_close_to_4)
    print(f"Added sample {sample_close_to_4}. Clusters: {len(clustering_system.clusters)}")
    for j, c in enumerate(clustering_system.clusters):
        print(f"  Cluster {j} (mean: {mean(c)}): {list(c.samples)}")

    # Check if oldest samples were removed correctly
    # With P=2, each cluster should have max 2 samples.
    # E.g., for cluster 0, if [0, 0.5] was first, then [0.1, 0.6] was added,
    # and it was already size 2, then [0, 0.5] should be removed.