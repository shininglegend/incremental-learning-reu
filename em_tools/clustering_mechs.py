# Handles the clusters themselves - assigning and removing as needed
import torch
import pandas
from collections import deque

DEBUG = False

triggered = set()
dtriggered = lambda s: triggered.add(s) if DEBUG else None
# Helper function that allows conditional function execution based on flag above
debug = lambda f, *args, **kwargs: f(*args, **kwargs) if DEBUG else None


class Cluster:
    """Represents a single cluster in the clustering mechanism."""

    def __init__(self, initial_sample: torch.Tensor, initial_label=None):
        dtriggered("cl init triggered")

        self.samples = deque([initial_sample])  # Stores samples in insertion order
        self.labels = (
            deque([initial_label]) if initial_label is not None else deque([None])
        )  # Stores labels in insertion order
        self.mean = initial_sample.clone().detach()
        self.sum_samples = initial_sample.clone().detach()  # To efficiently update mean

    def get_sample_or_samples(self):
        """
        If this cluster is of size one, returns the sample inside.
        Otherwise, returns the samples as a list or [].
        Always returns the label stored alongside if present

        Returns:
            : The stored samples
        """
        if len(self.samples) == 1:
            return (self.samples[0], self.labels[0])
        return (list(self.samples), list(self.labels))

    def add_sample(self, sample: torch.Tensor, label=None):
        """Adds a new sample to the cluster and updates its mean."""
        dtriggered("cl add_sample triggered")

        self.samples.append(sample)
        self.labels.append(label)
        self.sum_samples += sample.clone().detach()
        self.mean = self.sum_samples / len(self.samples)

    def remove_one(self):
        """
        Removes a sample from the cluster and updates its mean.
        """
        dtriggered("cl remove_one triggered")

        return self.remove_based_on_mean()

    def remove_based_on_mean(self):
        """
        Removes the sample furthest from the mean, excluding the newest sample.
        """
        dtriggered("cl remove_based_on_mean triggered")

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
        Removes a sample from the cluster and updates its mean.
        Currently removes the oldest sample.
        """
        dtriggered("cl remove_oldest triggered")

        if len(self.samples) > 0:
            oldest_sample = self.samples.popleft()
            self.labels.popleft()  # Also remove the corresponding label
            self.sum_samples -= oldest_sample
            if len(self.samples) > 0:
                self.mean = self.sum_samples / len(self.samples)
            else:
                self.mean = torch.zeros_like(self.mean)  # If cluster becomes empty

    def __str__(self):
        return f"Cluster with mean {self.mean} and labeled samples: " + "\n".join(
            [
                (f"({sample}, {label})")
                for sample, label in zip(self.samples, self.labels)
            ]
        )

    def __len__(self):
        return len(self.samples)


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
        dtriggered("clm init triggered")

        self.clusters: list[Cluster] = []  # List of Cluster objects
        self.Q = Q  # Max number of clusters
        self.P = P  # Max cluster size
        self.dimensionality_reducer = dimensionality_reducer

        # Sanity checks
        assert self.Q > 0
        assert self.P > 0

    def add(self, z: torch.Tensor, label=None):
        """
        Adds a sample z to the appropriate cluster or forms a new one.

        Args:
            z (torch.Tensor): The sample (e.g., activation vector) to add.
            label: Optional label associated with the sample (does not affect clustering).
        """
        dtriggered("clm add triggered")

        assert len(z.shape) == 1, "Sample should only have one axis."

        # Apply dimensionality reduction if configured
        if self.dimensionality_reducer is not None:
            if not self.dimensionality_reducer.fitted:
                raise ValueError(
                    "Dimensionality reducer must be fitted before adding samples. Call fit_reducer() first."
                )
            z = self.dimensionality_reducer.transform(z)

        # If we're at 0 samples, or at 1 sample and there's space for a second one, add it
        if len(self.clusters) == 0 or (len(self.clusters) == 1 and self.Q > 1):
            return self._add_new_cluster(z, label)

        # Find the closest cluster
        (nearest_cluster_idx_to_new, dist_to_new) = self._find_closest_cluster(z)

        # See if we can re-assign clusters to make this a new mean
        # Logic: If any cluster of size 1 is closer to another cluster than this sample is to it,
        # merge the sample from that cluster into whatever it's closer to, and remove the old cluster,
        # replacing it with a new sample centered on this new sample

        # Start by finding the current two closest clusters to each other, for each cluster of size 1
        # This is n^2 where n is the number of clusters (Q), so should probably be optimized
        min_dist = math.inf
        cluster_idx_to_merge, cluster_idx_to_merge_into = None, None
        for i in range(len(self.clusters)):
            if len(self.clusters[i]) > 1:
                # We don't want to reassign from a cluster that's bigger than size 1
                continue
            for j in range(i+1, len(self.clusters)):
                if i == j:
                    continue  # Ignore itself, obv
                dist = self._dist_between(i, j)
                if min_dist > dist:
                    min_dist = dist
                    # Don't flip the letters! Otherwise, you could be merging a cluster of size > 1
                    cluster_idx_to_merge = i
                    cluster_idx_to_merge_into = j

        if min_dist == math.inf or  min_dist >= dist_to_new:
            # No mergable clusters found
            debug(print, f"Kept! {min_dist} >= {dist_to_new}")
            if len(self.clusters) < self.Q:
                # Add a new cluster if possible
                return self._add_new_cluster(z, label)
            else:
                return self._add_to_cluster(nearest_cluster_idx_to_new, z, label)

        # Merge the cluster, then add the new sample as a new cluster
        self._add_to_cluster(
            cluster_idx_to_merge_into,
            self.clusters[cluster_idx_to_merge].samples.pop(),
            self.clusters[cluster_idx_to_merge].labels.pop(),
        )

        debug(print, f"Overwrote! {dist_to_new} > {min_dist}")
        self.clusters[cluster_idx_to_merge] = Cluster(z, label)

    def _dist_between(self, cluster_a_indx, cluster_b_indx):
        """Returns the distance between two clusters's means"""
        return torch.linalg.norm(
            self.clusters[cluster_a_indx].mean - self.clusters[cluster_b_indx].mean
        )

    def _find_closest_cluster(self, z, ignore=None):
        """This finds the closest cluster for a particular sample. Does not add it

        Args:
            z (Tensor): Sample to find the closest cluster for
            ignore (index): index of cluster to ignore if given

        Returns:
            (index, distance): Returns the index and distance to the closest node
        """
        # If the number of clusters has reached Q, find the closest cluster
        # based on Euclidean distance to its mean.
        min_distance = float("inf")
        closest_cluster_idx = -1

        for i, cluster in enumerate(self.clusters):
            if ignore is not None and i == ignore:
                continue
            # Calculate Euclidean distance
            distance = torch.linalg.norm(z - cluster.mean)
            if distance < min_distance:
                min_distance = distance
                closest_cluster_idx = i
        return (closest_cluster_idx, min_distance)

    def _add_new_cluster(self, z, label=None):
        """This adds a new cluster with the given sample and label

        Args:
            z (Tensor): Sample to be added
            label (, optional): Label to assign to the sample. Defaults to None.
        """
        new_cluster = Cluster(z, label)
        self.clusters.append(new_cluster)

    def _add_to_cluster(self, cluster_index, z, label=None):
        """Adds a sample to a given cluster, removing one from that cluster if needed to make space

        Args:
            cluster_index (int): Index of the cluster
            z (Tensor): Sample to add
            label (_type_, optional): Optional label of sample. Defaults to None.
        """
        # Add z to the identified closest cluster
        q_star: Cluster = self.clusters[cluster_index]
        q_star.add_sample(z, label)

        # If the cluster size exceeds P, remove the oldest sample
        if len(q_star.samples) > self.P:
            q_star.remove_one()

    def fit_reducer(self, z_list):
        """
        Fit dimensionality reducer on a set of samples. Must be called before adding samples if reducer is configured.

        Args:
            z_list (list or torch.Tensor): List of samples to fit reducer on
        """
        dtriggered("clm fit_reducer triggered")

        if self.dimensionality_reducer.fitted:
            return
        if self.dimensionality_reducer is None:
            return

        self.dimensionality_reducer.fit(z_list)

    def transform(self, z):
        """
        Apply dimensionality reduction transformation to external data (e.g., test data).

        Args:
            z (torch.Tensor): Sample or batch of samples to transform

        Returns:
            torch.Tensor: Transformed data
        """
        dtriggered("clm transform triggered")

        if self.dimensionality_reducer is None:
            return z

        if not self.dimensionality_reducer.fitted:
            raise ValueError(
                "Dimensionality reducer must be fitted before transforming data. Call fit_reducer() first."
            )

        return self.dimensionality_reducer.transform(z)

    def add_multi(self, z_list, labels=None):
        """
        This adds a list of samples to the system. TM

        Args:
            z_list (torch.Tensor): list of samples to add
            labels: Optional list of labels corresponding to samples
        """
        dtriggered("clm add_multi triggered")

        self.fit_reducer(z_list)
        if labels is not None:
            [self.add(z, label) for z, label in zip(z_list, labels)]
        else:
            [self.add(z) for z in z_list]

    def get_clusters_for_training(self):
        """Gets the samples currently stored in the clusters

        Returns:
            torch.Tensor: an array of samples currently stored in the clusters
        """
        dtriggered("clm get_clusters_for_training triggered")

        all_samples = []
        for cluster in self.clusters:
            for sample in cluster.samples:
                all_samples.append(sample)
        return torch.stack(all_samples) if all_samples else torch.tensor([])

    def get_clusters_with_labels(self):
        """Gets the samples and their labels currently stored in the clusters

        Returns:
            tuple: (torch.Tensor of samples, list of labels)
        """
        dtriggered("clm get_clusters_with_labels triggered")

        all_samples = []
        all_labels = []
        for i in range(len(self.clusters)):
            assert len(self.clusters[i].samples) == len(
                self.clusters[i].labels
            ), "Missing labels - spoof if needed"
            all_samples.extend(self.clusters[i].samples)
            all_labels.extend(self.clusters[i].labels)
        samples_array = torch.stack(all_samples) if all_samples else torch.tensor([])
        return samples_array, all_labels

    def visualize(self, title_postfix=""):
        """
        Show what's stored inside!
        """
        dtriggered("clm visualize triggered")
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

        # Convert to tensor
        X = torch.stack(all_samples)

        # Apply PCA to reduce to 3 dimensions if needed
        if X.shape[1] > 3:
            pca = PCA(n_components=3)
            X_reduced = pca.fit_transform(X.numpy())
            X_reduced = torch.tensor(X_reduced, dtype=torch.float32)
            # Create column names for the PCA components
            x_col, y_col, z_col = "PC1", "PC2", "PC3"
        else:
            X_reduced = X
            # Pad with zeros if less than 3 dimensions
            if X_reduced.shape[1] == 1:
                X_reduced = torch.column_stack(
                    [
                        X_reduced,
                        torch.zeros(X_reduced.shape[0]),
                        torch.zeros(X_reduced.shape[0]),
                    ]
                )
            elif X_reduced.shape[1] == 2:
                X_reduced = torch.column_stack(
                    [X_reduced, torch.zeros(X_reduced.shape[0])]
                )
            x_col, y_col, z_col = "X", "Y", "Z"

        # Create DataFrame for plotly
        df = pandas.DataFrame(
            {
                x_col: X_reduced[:, 0].numpy(),
                y_col: X_reduced[:, 1].numpy(),
                z_col: X_reduced[:, 2].numpy(),
                "Cluster": cluster_labels,
                "Label": true_labels,
            }
        )

        # Create 3D scatter plot
        title = "Cluster Visualization (Color=Cluster, Symbol=Label)"
        title += " | " + title_postfix if title_postfix != "" else ""
        fig = px.scatter_3d(
            df,
            x=x_col,
            y=y_col,
            z=z_col,
            color="Cluster",
            symbol="Label",
            title=title,
            hover_data={"Cluster": True, "Label": True},
        )

        fig.show()


if __name__ == "__main__":
    print("Testing Cluster Storage")
    NUM_SAMPLES = 20
    CLUSTERS = 3
    MAX_PER_CLUSTER = 3
    VISUALIZE = False
    if VISUALIZE:
        import time

    storage = ClusteringMechanism(Q=CLUSTERS, P=MAX_PER_CLUSTER)
    # Generate CLUSTERS+1 clusters with outliers
    import random, math

    # Set seeds for consistent generation
    torch.manual_seed(42)
    random.seed(42)

    samples = []
    # Generate
    samples_per_cluster = math.floor(NUM_SAMPLES / 3)
    for _ in range(samples_per_cluster):
        # Cluster 1: around [10, 10, 10]
        samples.append(
            torch.tensor([10, 10, 10], dtype=torch.float32) + torch.randn(3) * 2
        )
        # Cluster 2: around [50, 50, 50]
        samples.append(
            torch.tensor([50, 50, 50], dtype=torch.float32) + torch.randn(3) * 2
        )
        # Cluster 3: around [90, 90, 90]
        samples.append(
            torch.tensor([90, 90, 90], dtype=torch.float32) + torch.randn(3) * 2
        )
    # Outliers
    for _ in range(NUM_SAMPLES - (samples_per_cluster * 3)):
        samples.append(
            torch.tensor([25, 75, 25], dtype=torch.float32) + torch.randn(3) * 3
        )
    assert NUM_SAMPLES == len(samples), "Did not get the expected number of samples"

    # Shuffle samples
    random.shuffle(samples)

    print(storage.get_clusters_with_labels())
    for i, sample in enumerate(samples):
        storage.add(sample, label=i)
        print(storage.get_clusters_with_labels())

        if VISUALIZE:
            storage.visualize(f"After sample {i}")
            # It is recommended to set a breakpoint on the next line when visualizing
            time.sleep(1)

    if len(storage.clusters) == 0:
        raise Exception("No samples successfully added")

    [print(f"\n{i+1}:\n {storage.clusters[i]}\n") for i in range(len(storage.clusters))]
    print(sorted(list(triggered))) if len(triggered) > 0 else None
    storage.visualize()

    # Visualize samples in 3D
    import plotly.graph_objects as go

    sample_array = torch.stack(samples).numpy()
    colors = (
        ["red"] * samples_per_cluster
        + ["blue"] * samples_per_cluster
        + ["green"] * samples_per_cluster
        + ["black"] * (NUM_SAMPLES - samples_per_cluster)
    )
    random.shuffle(colors)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=sample_array[:, 0],
                y=sample_array[:, 1],
                z=sample_array[:, 2],
                mode="markers",
                marker=dict(size=8, color=colors, opacity=0.8),
                text=[f"Sample {i}" for i in range(len(samples))],
            )
        ]
    )

    fig.update_layout(
        title="Generated Test Samples (3 Clusters + Outliers)",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    )
    if VISUALIZE:
        fig.show()
