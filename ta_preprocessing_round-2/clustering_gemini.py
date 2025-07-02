# Initial Code by gemini. 1x2 matrix per item
import numpy as np
import pandas
from collections import deque

class Cluster:
    """Represents a single cluster in the clustering mechanism."""
    def __init__(self, initial_sample, initial_label=None):
        self.samples = deque([initial_sample]) # Stores samples in insertion order
        self.labels = deque([initial_label]) if initial_label is not None else deque([None]) # Stores labels in insertion order
        self.mean = np.array(initial_sample)
        self.sum_samples = np.array(initial_sample) # To efficiently update mean

    def add_sample(self, sample, label=None):
        """Adds a new sample to the cluster and updates its mean."""
        self.samples.append(sample)
        self.labels.append(label)
        self.sum_samples += np.array(sample)
        self.mean = self.sum_samples / len(self.samples)

    def remove_one(self):
        """
        Removes a sample from the cluster and updates its mean.
        """
        return self.remove_oldest()


    def remove_based_on_mean(self):
        """
        Removes the sample furthest from the mean, excluding the newest sample.
        """
        if len(self.samples) <= 1:
            return self.remove_oldest()

        # Find sample furthest from mean, excluding newest (last) sample
        max_distance = -1
        furthest_idx = 0

        # Only consider samples except the newest (last one)
        for i in range(len(self.samples) - 1):
            sample = self.samples[i]
            distance = np.linalg.norm(np.array(sample) - self.mean)
            if distance > max_distance:
                max_distance = distance
                furthest_idx = i

        # Remove the furthest sample and its label
        removed_sample = self.samples[furthest_idx]
        removed_label = self.labels[furthest_idx]

        # Convert deque to list for index-based removal
        samples_list = list(self.samples)
        labels_list = list(self.labels)

        samples_list.pop(furthest_idx)
        labels_list.pop(furthest_idx)

        # Convert back to deque
        self.samples = deque(samples_list)
        self.labels = deque(labels_list)

        # Update sum and mean
        self.sum_samples -= np.array(removed_sample)
        if len(self.samples) > 0:
            self.mean = self.sum_samples / len(self.samples)
        else:
            self.mean = np.zeros_like(self.mean)

        return removed_sample, removed_label


    def remove_oldest(self):
        """
        Removes a sample from the cluster and updates its mean.
        Currently removes the oldest sample.
        """
        if len(self.samples) > 0:
            oldest_sample = self.samples.popleft()
            self.labels.popleft() # Also remove the corresponding label
            self.sum_samples -= np.array(oldest_sample)
            if len(self.samples) > 0:
                self.mean = self.sum_samples / len(self.samples)
            else:
                self.mean = np.zeros_like(self.mean) # If cluster becomes empty

    def __str__(self):
        return f"""Cluster with mean {self.mean} and samples {self.samples}"""

class ClusteringMechanism:
    """Implements the clustering mechanism described in Algorithm 3."""

    def __init__(self, Q=100, P=3, dimensionality_reducer=None):
        """
        Initializes the clustering mechanism.

        Args:
            Q (int): Maximum number of clusters.
            P (int): Maximum size of each cluster.
            dimensionality_reducer (DimensionalityReducer): Dimensionality reduction method. If None, no reduction is applied.
        """
        self.clusters = []  # List of Cluster objects
        self.Q = Q          # Max number of clusters
        self.P = P          # Max cluster size
        self.dimensionality_reducer = dimensionality_reducer

    def add(self, z:np.ndarray, label=None):
        """
        Adds a sample z to the appropriate cluster or forms a new one.

        Args:
            z (np.ndarray): The sample (e.g., activation vector) to add.
            label: Optional label associated with the sample (does not affect clustering).
        """
        assert len(z.shape) == 1, "Sample should only have one axis."

        # Apply dimensionality reduction if configured
        if self.dimensionality_reducer is not None:
            if not self.dimensionality_reducer.fitted:
                raise ValueError("Dimensionality reducer must be fitted before adding samples. Call fit_reducer() first.")
            z = self.dimensionality_reducer.transform(z)
        # if z is a set of samples, add it one by one
        if len(self.clusters) < self.Q:
            # If the number of clusters is less than Q, create a new cluster
            # and add z to it.
            new_cluster = Cluster(z, label)
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
            q_star.add_sample(z, label)

            # If the cluster size exceeds P, remove the oldest sample
            if len(q_star.samples) > self.P:
                q_star.remove_one()


    def fit_reducer(self, z_list):
        """
        Fit dimensionality reducer on a set of samples. Must be called before adding samples if reducer is configured.

        Args:
            z_list (list or np.ndarray): List of samples to fit reducer on
        """
        if self.dimensionality_reducer.fitted:
            return
        if self.dimensionality_reducer is None:
            return

        self.dimensionality_reducer.fit(z_list)

    def transform(self, z):
        """
        Apply dimensionality reduction transformation to external data (e.g., test data).

        Args:
            z (np.ndarray): Sample or batch of samples to transform

        Returns:
            np.ndarray: Transformed data
        """
        if self.dimensionality_reducer is None:
            return z

        if not self.dimensionality_reducer.fitted:
            raise ValueError("Dimensionality reducer must be fitted before transforming data. Call fit_reducer() first.")

        return self.dimensionality_reducer.transform(z)

    def add_multi(self, z_list, labels=None):
        """
        This adds a list of samples to the system. TM

        Args:
            z_list (ndp_array): list of samples to add
            labels: Optional list of labels corresponding to samples
        """
        self.fit_reducer(z_list)
        if labels is not None:
            [self.add(z, label) for z, label in zip(z_list, labels)]
        else:
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
        return np.array(all_samples) if all_samples else np.array([])

    def get_clusters_with_labels(self):
        """Gets the samples and their labels currently stored in the clusters

        Returns:
            tuple: (np.ndarray of samples, list of labels)
        """
        all_samples = []
        all_labels = []
        for cluster in self.clusters:
            for sample, label in zip(cluster.samples, cluster.labels):
                all_samples.append(sample)
                all_labels.append(label)
        samples_array = np.array(all_samples) if all_samples else np.array([])
        return samples_array, all_labels

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

        # Collect all samples with cluster labels and true labels
        all_samples = []
        cluster_labels = []
        true_labels = []

        for cluster_idx, cluster in enumerate(self.clusters):
            for sample, label in zip(cluster.samples, cluster.labels):
                all_samples.append(sample)
                cluster_labels.append(f"Cluster {cluster_idx}")
                true_labels.append(str(label) if label is not None else "None")

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
            'Cluster': cluster_labels,
            'Label': true_labels
        })

        # Create 3D scatter plot
        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col,
                           color='Cluster',
                           symbol='Label',
                           title='Cluster Visualization (Color=Cluster, Symbol=Label)',
                           hover_data={'Cluster': True, 'Label': True})

        fig.show()
