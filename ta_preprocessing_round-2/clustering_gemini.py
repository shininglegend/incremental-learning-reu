# no numpy conversions
import torch
from collections import deque


class Cluster:
    """Represents a single cluster in the clustering mechanism."""

    def __init__(self, initial_sample, initial_label=None, device='cpu'):
        self.device = device

        # Ensure initial_sample is a tensor on the correct device
        if not isinstance(initial_sample, torch.Tensor):
            initial_sample = torch.tensor(initial_sample, dtype=torch.float32, device=device)
        else:
            initial_sample = initial_sample.to(device)

        self.samples = deque([initial_sample])  # Stores samples in insertion order
        self.labels = deque([initial_label]) if initial_label is not None else deque([None])
        self.mean = initial_sample.clone()
        self.sum_samples = initial_sample.clone()  # To efficiently update mean

    def add_sample(self, sample, label=None):
        """Adds a new sample to the cluster and updates its mean."""
        # Ensure sample is a tensor on the correct device
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32, device=self.device)
        else:
            sample = sample.to(self.device)

        self.samples.append(sample)
        self.labels.append(label)
        self.sum_samples += sample
        self.mean = self.sum_samples / len(self.samples)

    def remove_one(self):
        """
        Removes a sample from the cluster and updates its mean.
        """
        return self.remove_based_on_mean()

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
            distance = torch.norm(sample - self.mean)
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
        self.sum_samples -= removed_sample
        if len(self.samples) > 0:
            self.mean = self.sum_samples / len(self.samples)
        else:
            self.mean = torch.zeros_like(self.mean)

        return removed_sample, removed_label

    def remove_oldest(self):
        """
        Removes the oldest sample from the cluster and updates its mean.
        """
        if len(self.samples) > 0:
            oldest_sample = self.samples.popleft()
            oldest_label = self.labels.popleft()
            self.sum_samples -= oldest_sample
            if len(self.samples) > 0:
                self.mean = self.sum_samples / len(self.samples)
            else:
                self.mean = torch.zeros_like(self.mean)
            return oldest_sample, oldest_label
        return None, None

    def __str__(self):
        return f"""Cluster with mean {self.mean} and {len(self.samples)} samples"""


class ClusteringMechanism:
    """Implements the clustering mechanism described in Algorithm 3."""

    def __init__(self, Q=100, P=3, device='cpu'):
        """
        Initializes the clustering mechanism.

        Args:
            Q (int): Maximum number of clusters.
            P (int): Maximum size of each cluster.
            device: PyTorch device to use for tensor operations.
        """
        self.clusters = []  # List of Cluster objects
        self.Q = Q  # Max number of clusters
        self.P = P  # Max cluster size
        self.device = device

    def add(self, z, label=None):
        """
        Adds a sample z to the appropriate cluster or forms a new one.

        Args:
            z (torch.Tensor): The sample (e.g., activation vector) to add.
            label: Optional label associated with the sample (does not affect clustering).
        """
        # Ensure z is a tensor on the correct device
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.float32, device=self.device)
        else:
            z = z.to(self.device)

        # Ensure z is 1D
        if len(z.shape) != 1:
            z = z.flatten()

        if len(self.clusters) < self.Q:
            # If the number of clusters is less than Q, create a new cluster
            new_cluster = Cluster(z, label, device=self.device)
            self.clusters.append(new_cluster)
        else:
            # Find the closest cluster based on Euclidean distance to its mean
            min_distance = float('inf')
            closest_cluster_idx = -1

            for i, cluster in enumerate(self.clusters):
                # Calculate Euclidean distance using PyTorch
                distance = torch.norm(z - cluster.mean)
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster_idx = i

            # Add z to the identified closest cluster
            q_star = self.clusters[closest_cluster_idx]
            q_star.add_sample(z, label)

            # If the cluster size exceeds P, remove the oldest sample
            if len(q_star.samples) > self.P:
                q_star.remove_one()

    def add_multi(self, z_list, labels=None):
        """
        Adds a list of samples to the system.

        Args:
            z_list: List of samples to add (should be tensors or convertible to tensors)
            labels: Optional list of labels corresponding to samples
        """
        if labels is not None:
            for z, label in zip(z_list, labels):
                self.add(z, label)
        else:
            for z in z_list:
                self.add(z)

    def get_clusters_for_training(self):
        """Gets the samples currently stored in the clusters

        Returns:
            torch.Tensor: A tensor of samples currently stored in the clusters
        """
        all_samples = []
        for cluster in self.clusters:
            for sample in cluster.samples:
                all_samples.append(sample)

        if all_samples:
            return torch.stack(all_samples)
        else:
            return torch.empty(0, device=self.device)

    def get_clusters_with_labels(self):
        """Gets the samples and their labels currently stored in the clusters

        Returns:
            tuple: (list of tensor samples, list of labels)
        """
        all_samples = []
        all_labels = []
        for cluster in self.clusters:
            for sample, label in zip(cluster.samples, cluster.labels):
                all_samples.append(sample)
                all_labels.append(label)

        return all_samples, all_labels

    def visualize(self):
        """
        Show what's stored inside using tensor operations!
        """
        # Show each cluster in its own color
        try:
            import plotly.express as px
            import pandas as pd
            from sklearn.decomposition import PCA
        except ImportError:
            print("Visualization requires plotly, pandas, and sklearn")
            return

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

        # Stack tensors and convert to numpy for sklearn PCA
        X_tensor = torch.stack(all_samples)
        X_numpy = X_tensor.cpu().numpy()  # Only convert to numpy for visualization

        # Apply PCA to reduce to 3 dimensions if needed
        if X_numpy.shape[1] > 3:
            pca = PCA(n_components=3)
            X_reduced = pca.fit_transform(X_numpy)
            x_col, y_col, z_col = 'PC1', 'PC2', 'PC3'
        else:
            X_reduced = X_numpy
            # Pad with zeros if less than 3 dimensions
            if X_reduced.shape[1] == 1:
                X_reduced = torch.cat([
                    torch.from_numpy(X_reduced),
                    torch.zeros(X_reduced.shape[0], 2)
                ], dim=1).numpy()
            elif X_reduced.shape[1] == 2:
                X_reduced = torch.cat([
                    torch.from_numpy(X_reduced),
                    torch.zeros(X_reduced.shape[0], 1)
                ], dim=1).numpy()
            x_col, y_col, z_col = 'X', 'Y', 'Z'

        # Create DataFrame for plotly
        df = pd.DataFrame({
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