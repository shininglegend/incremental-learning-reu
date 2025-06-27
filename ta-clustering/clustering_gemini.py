# Initial Code by gemini. 1x2 matrix per item
import numpy as np
import pandas
from collections import deque
from sklearn.decomposition import PCA

class Cluster:
    """Represents a single cluster in the clustering mechanism."""
    def __init__(self, initial_sample):
        self.samples = deque([initial_sample]) # Stores samples in insertion order
        self.mean = np.array(initial_sample)
        self.sum_samples = np.array(initial_sample) # To efficiently update mean

    def add_sample(self, sample):
        """Adds a new sample to the cluster and updates its mean."""
        self.samples.append(sample)
        self.sum_samples += np.array(sample)
        self.mean = self.sum_samples / len(self.samples)

    def remove_oldest(self):
        """Removes the oldest sample from the cluster and updates its mean."""
        if len(self.samples) > 0:
            oldest_sample = self.samples.popleft()
            self.sum_samples -= np.array(oldest_sample)
            if len(self.samples) > 0:
                self.mean = self.sum_samples / len(self.samples)
            else:
                self.mean = np.zeros_like(self.mean) # If cluster becomes empty

    def __str__(self):
        return f"""Cluster with mean {self.mean} and samples {self.samples}"""

class ClusteringMechanism:
    """Implements the clustering mechanism described in Algorithm 3."""

    def __init__(self, Q=100, P=3, pca_components=None):
        """
        Initializes the clustering mechanism.

        Args:
            Q (int): Maximum number of clusters.
            P (int): Maximum size of each cluster.
            pca_components (int): Number of PCA components to reduce to. If None, no PCA is applied.
        """
        self.clusters = []  # List of Cluster objects
        self.Q = Q          # Max number of clusters
        self.P = P          # Max cluster size
        self.pca_components = pca_components
        self.pca = PCA(n_components=pca_components) if pca_components else None
        self.pca_fitted = False

    def add(self, z:np.ndarray):
        """
        Adds a sample z to the appropriate cluster or forms a new one.

        Args:
            z (np.ndarray): The sample (e.g., activation vector) to add.
        """
        assert len(z.shape) == 1, "Sample should only have one axis."

        # Apply PCA if configured
        if self.pca is not None:
            if not self.pca_fitted:
                raise ValueError("PCA must be fitted before adding samples. Call fit_pca() first.")
            z_reshaped = z.reshape(1, -1)  # PCA expects 2D input
            z = self.pca.transform(z_reshaped).flatten()
        # if z is a set of samples, add it one by one
        if len(self.clusters) < self.Q:
            # If the number of clusters is less than Q, create a new cluster
            # and add z to it.
            new_cluster = Cluster(z)
            self.clusters.append(new_cluster)
        else:
            # If the number of clusters has reached Q, find the closest cluster
            # based on Euclidean distance to its mean.
            min_distance = float('inf')
            closest_cluster_idx = -1

            for i, cluster in enumerate(self.clusters):
                # Calculate Euclidean distance
                distance = np.linalg.norm(z - cluster.mean)
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster_idx = i

            # Add z to the identified closest cluster
            q_star = self.clusters[closest_cluster_idx]
            q_star.add_sample(z)

            # If the cluster size exceeds P, remove the oldest sample
            if len(q_star.samples) > self.P:
                q_star.remove_oldest()


    def fit_pca(self, z_list):
        """
        Fit PCA on a set of samples. Must be called before adding samples if PCA is configured.

        Args:
            z_list (list or np.ndarray): List of samples to fit PCA on
        """
        if self.pca is None:
            return

        if isinstance(z_list, list):
            X = np.array(z_list)
        else:
            X = z_list

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        self.pca.fit(X)
        self.pca_fitted = True

    def add_multi(self, z_list):
        """
        This adds a list of samples to the system. TM

        Args:
            z_list (ndp_array): list of samples to add
        """
        [self.add(z) for z in z_list]

    def get_clusters_for_training(self) -> np.ndarray:
        """Gets the samples currently stored in the clusters

        Returns:
            np.ndarray: an array of samples currently stored in the clusters
        """
        all_samples = []
        for cluster in self.clusters:
            for sample in cluster.samples:
                all_samples.append(sample)
        return np.array(all_samples) if all_samples else np.array([]).reshape(0, -1)

    def visualize(self):
        """
        Show what's stored inside!
        """
        # Show each cluster in it's own color
        import plotly.express as px
        from sklearn.decomposition import PCA

        if not self.clusters:
            print("No clusters to visualize")
            return

        # Collect all samples with cluster labels
        all_samples = []
        cluster_labels = []

        for cluster_idx, cluster in enumerate(self.clusters):
            for sample in cluster.samples:
                all_samples.append(sample)
                cluster_labels.append(f"Cluster {cluster_idx}")

        if not all_samples:
            print("No samples to visualize")
            return

        # Convert to numpy array
        X = np.array(all_samples)

        # Apply PCA to reduce to 3 dimensions if needed
        if X.shape[1] > 3:
            pca = PCA(n_components=3)
            X_reduced = pca.fit_transform(X)
            # Create column names for the PCA components
            x_col, y_col, z_col = 'PC1', 'PC2', 'PC3'
        else:
            X_reduced = X
            # Pad with zeros if less than 3 dimensions
            if X_reduced.shape[1] == 1:
                X_reduced = np.column_stack([X_reduced, np.zeros(X_reduced.shape[0]), np.zeros(X_reduced.shape[0])])
            elif X_reduced.shape[1] == 2:
                X_reduced = np.column_stack([X_reduced, np.zeros(X_reduced.shape[0])])
            x_col, y_col, z_col = 'X', 'Y', 'Z'

        # Create DataFrame for plotly
        df = pandas.DataFrame({
            x_col: X_reduced[:, 0],
            y_col: X_reduced[:, 1],
            z_col: X_reduced[:, 2],
            'Cluster': cluster_labels
        })

        # Create 3D scatter plot
        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col,
                           color='Cluster',
                           title='Cluster Visualization',
                           hover_data={'Cluster': True})

        fig.show()




# Example Usage for ClusteringMechanism:
if __name__ == '__main__':
    clustering_system = ClusteringMechanism(Q=3, P=2) # Smaller Q, P for demonstration

    print("Adding first 5 samples to initialize clusters:")
    for i in range(5):
        sample = np.array([i, i + 0.5])
        clustering_system.add(sample)
        print(f"Added sample {sample}. Clusters: {len(clustering_system.clusters)}")
        for j, c in enumerate(clustering_system.clusters):
            print(f"  Cluster {j} (mean: {c.mean}): {list(c.samples)}")

    print("\nAdding more samples, testing cluster size limit and mean update:")
    # These samples should go into existing clusters, triggering removal of oldest
    sample_close_to_0 = np.array([0.1, 0.6])
    clustering_system.add(sample_close_to_0)
    print(f"Added sample {sample_close_to_0}. Clusters: {len(clustering_system.clusters)}")
    for j, c in enumerate(clustering_system.clusters):
        print(f"  Cluster {j} (mean: {c.mean}): {list(c.samples)}")

    sample_close_to_4 = np.array([4.1, 4.6])
    clustering_system.add(sample_close_to_4)
    print(f"Added sample {sample_close_to_4}. Clusters: {len(clustering_system.clusters)}")
    for j, c in enumerate(clustering_system.clusters):
        print(f"  Cluster {j} (mean: {c.mean}): {list(c.samples)}")

    # Check if oldest samples were removed correctly
    # With P=2, each cluster should have max 2 samples.
    # E.g., for cluster 0, if [0, 0.5] was first, then [0.1, 0.6] was added,
    # and it was already size 2, then [0, 0.5] should be removed.
