# Handles the clusters themselves - assigning and removing as needed
import torch
import pandas
from collections import deque

DEBUG = False

triggered = set()
dsave = lambda s: triggered.add(s) if DEBUG else None
dprint = lambda *args: print(*args) if DEBUG else None


# Distance heuristic
def distance_h(*args, **kwargs):
    return torch.linalg.vector_norm(*args, **kwargs)


class Cluster:
    """Represents a single cluster in the clustering mechanism."""

    def __init__(
        self, initial_sample: torch.Tensor, initial_label=None, initial_task_id=None
    ):
        dsave("cl init triggered")

        self.samples = deque([initial_sample])  # Stores samples in insertion order
        self.labels = (
            deque([initial_label]) if initial_label is not None else deque([None])
        )  # Stores labels in insertion order
        self.task_ids = deque([initial_task_id])  # Stores task IDs for each sample
        self.insertion_order = deque(
            [0]
        )  # Stores insertion order (age) for each sample
        self.next_insertion_id = 1  # Counter for next insertion
        self.mean = initial_sample.clone().detach()
        self.sum_samples = initial_sample.clone().detach()  # To efficiently update mean

    def __len__(self):
        return len(self.samples)

    def add_sample(self, sample: torch.Tensor, label=None, task_id=None):
        """Adds a new sample to the cluster and updates its mean."""
        dsave("cl add_sample triggered")

        self.samples.append(sample)
        self.labels.append(label)
        self.task_ids.append(task_id)
        self.insertion_order.append(self.next_insertion_id)
        self.next_insertion_id += 1
        self.sum_samples += sample.clone().detach()
        self.mean = self.sum_samples / len(self.samples)

    def remove_one(self):
        """
        Removes a sample from the cluster and updates its mean.
        """
        dsave("cl remove_one triggered")

        return self.remove_oldest()

    def remove_based_on_mean(self):
        """
        Removes the sample furthest from the mean, excluding the newest sample.
        """
        dsave("cl remove_based_on_mean triggered")

        if len(self.samples) <= 1:
            return self.remove_oldest()

        # Find sample furthest from mean, excluding newest (last) sample
        max_distance = -1
        furthest_idx = 0

        # Only consider samples except the newest (last one)
        for i in range(len(self.samples) - 1):
            sample = self.samples[i]
            distance = distance_h(sample - self.mean)
            if distance > max_distance:
                max_distance = distance
                furthest_idx = i

        # Remove the furthest sample and its metadata
        removed_sample = self.samples[furthest_idx]
        removed_label = self.labels[furthest_idx]

        # Convert deque to list for index-based removal
        samples_list = list(self.samples)
        labels_list = list(self.labels)
        task_ids_list = list(self.task_ids)
        insertion_order_list = list(self.insertion_order)

        samples_list.pop(furthest_idx)
        labels_list.pop(furthest_idx)
        task_ids_list.pop(furthest_idx)
        insertion_order_list.pop(furthest_idx)

        # Convert back to deque
        self.samples = deque(samples_list)
        self.labels = deque(labels_list)
        self.task_ids = deque(task_ids_list)
        self.insertion_order = deque(insertion_order_list)

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
        dsave("cl remove_oldest triggered")

        if len(self.samples) > 0:
            oldest_sample = self.samples.popleft()
            self.labels.popleft()  # Also remove the corresponding label
            self.task_ids.popleft()  # Also remove the corresponding task_id
            self.insertion_order.popleft()  # Also remove the corresponding insertion_order
            self.sum_samples -= oldest_sample
            if len(self.samples) > 0:
                self.mean = self.sum_samples / len(self.samples)
            else:
                self.mean = torch.zeros_like(self.mean)  # If cluster becomes empty

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
        dsave("clm init triggered")

        self.clusters: list[Cluster] = []  # List of Cluster objects
        self.Q = Q  # Max number of clusters
        self.P = P  # Max cluster size
        self.max_size = Q * P
        self.dimensionality_reducer = dimensionality_reducer

    def __len__(self):
        return sum([len(cluster) for cluster in self.clusters])

    def add(self, z: torch.Tensor, label=None, task_id=None, timer=None):
        """
        Adds a sample z to the appropriate cluster or forms a new one.

        Args:
            z (torch.Tensor): The sample (e.g., activation vector) to add.
            label: Optional label associated with the sample (does not affect clustering).
            task_id: Optional task ID to track which task this sample came from.
        """
        dsave("clm add triggered")
        assert len(z.shape) == 1, "Sample should only have one axis."

        # Apply dimensionality reduction if configured
        if self.dimensionality_reducer is not None:
            if not self.dimensionality_reducer.fitted:
                raise ValueError(
                    "If provided, a dimensionality reducer must be fitted before adding samples. Call fit_reducer() first."
                )
            z = self.dimensionality_reducer.transform(z)

        if timer:
            timer.start("adding")
        self.add_allow_larger_clusters(z, label, task_id)
        if timer:
            timer.end("adding")

        # Change this to decide how to manage max cluster size
        # self._manage_by_cluster_size(new_cluster)
        if timer:
            timer.start("manage_pool")
        self._manage_by_pool_size(timer)
        if timer:
            timer.end("manage_pool")

        # Sanity checks
        if timer:
            timer.start("sanity")
        assert len(self) <= self.max_size, "Max pool size exceeded"
        assert len(self.clusters) <= self.Q, "Max number of clusters exceeded"
        if timer:
            timer.end("sanity")

    def add_normal(self, z, label, task_id):
        if len(self.clusters) < self.Q:
            # If the number of clusters is less than Q, create a new cluster
            # and add z to it.
            new_cluster = Cluster(z, label, task_id)
            self.clusters.append(new_cluster)
        else:
            # If the number of clusters has reached Q, find the closest cluster
            # based on Euclidean distance to its mean.
            (closest_cluster_idx, _) = self._find_closest_cluster(z)

            # Add z to the identified closest cluster
            self.clusters[closest_cluster_idx].add_sample(z)

    def add_allow_larger_clusters(self, z, label, task_id):
        # If there's space for another cluster, just add it
        if len(self.clusters) < self.Q:
            self.clusters.append(Cluster(z, label, task_id))

        # Otherwise, find the cluster using the following algorithm:
        # Find the closest cluster to this sample
        # If that cluster is further from the new sample than that cluster is to any other cluster,
        # simply merge it to it's closest cluster, not respecting Q, and then make the new sample it's own cluster
        # Then, run a second function to ensure this pool doesn't exceed max size
        # However, the max number of Clusters should always be less than P
        (closest_cluster_to_new_sample, dist_to_new_sample) = (
            self._find_closest_cluster(z)
        )
        (cluster_to_merge_from, cluster_to_merge_into, dist_between) = (
            self._find_two_closest_clusters()
        )
        if dist_to_new_sample <= dist_between:
            dprint(f"Kept: {dist_to_new_sample} <= {dist_between}")
            # Add the sample to it's closest cluster instead
            self.clusters[closest_cluster_to_new_sample].add_sample(z, label, task_id)
        else:
            dprint(f"Merged: {dist_to_new_sample} > {dist_between}")
            [
                self.clusters[cluster_to_merge_into].add_sample(z, label, task_id)
                for (z, label, task_id) in zip(
                    self.clusters[cluster_to_merge_from].samples,
                    self.clusters[cluster_to_merge_from].labels,
                    self.clusters[cluster_to_merge_from].task_ids,
                )
            ]
            self.clusters[cluster_to_merge_from] = Cluster(z, label, task_id)

    def _manage_by_cluster_size(self, new_idx):
        # If the cluster size exceeds P, remove the oldest sample
        if len(self.clusters[new_idx].samples) > self.P:
            self.clusters[new_idx].remove_one()

    def _manage_by_pool_size(self, timer=None):
        """This function removes a sample from the largest cluster
        if needed to remain under size restrictions
        """
        if timer:
            timer.start("manage_pool_inside")
        if len(self) <= self.max_size:
            if timer:
                timer.end("manage_pool_inside")
            return

        if timer:
            timer.start("B")
        largest_idx = max(
            range(len(self.clusters)), key=lambda i: len(self.clusters[i])
        )
        # print("\n", largest_idx, [len(cluster) for cluster in self.clusters])
        assert len(self.clusters[largest_idx]) > self.P
        if timer:
            timer.end("B")
        # dprint(f"Before: {self.clusters[largest_idx]}")
        # if DEBUG: self.visualize("Before")
        if timer:
            timer.start("C")
        self.clusters[largest_idx].remove_one()
        if timer:
            timer.end("C")
        # if DEBUG: self.visualize("After")
        # dprint(f"After: {self.clusters[largest_idx]}")
        assert len(self) <= self.max_size
        if timer:
            timer.end("manage_pool_inside")
        # Optional: Call it recursively in case multiple samples were added
        # self._manage_by_pool_size(new_idx, timer)

    def _find_closest_cluster(self, z):
        """Given a sample, finds the closest cluster based on mean in O(k) time
        where k is the number of clusters in the pool

        Returns:
            (int, tensor_float): index of closest cluster, dist to closest cluster
        """
        min_distance = float("inf")
        closest_cluster_idx = -1

        for i, cluster in enumerate(self.clusters):
            # Calculate distance
            distance = distance_h(z - cluster.mean)
            if distance < min_distance:
                min_distance = distance
                closest_cluster_idx = i
        return (closest_cluster_idx, min_distance)

    def _find_two_closest_clusters(self):

        min_distance = float("inf")
        cluster_a, cluster_b = -1, -1
        for i in range(len(self.clusters)):
            for j in range(i + 1, len(self.clusters)):
                distance = distance_h(self.clusters[i].mean - self.clusters[j].mean)
                assert distance > 0, "Distance must be > 0"
                if distance < min_distance:
                    min_distance = distance
                    # The following two assignments are a bit arbitrary
                    cluster_a = i
                    cluster_b = j
        return (cluster_a, cluster_b, min_distance)

    def fit_reducer(self, z_list):
        """
        Fit dimensionality reducer on a set of samples. Must be called before adding samples if reducer is configured.

        Args:
            z_list (list or torch.Tensor): List of samples to fit reducer on
        """
        dsave("clm fit_reducer triggered")

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
        dsave("clm transform triggered")

        if self.dimensionality_reducer is None:
            return z

        if not self.dimensionality_reducer.fitted:
            raise ValueError(
                "Dimensionality reducer must be fitted before transforming data. Call fit_reducer() first."
            )

        return self.dimensionality_reducer.transform(z)

    def add_multi(self, z_list, labels=None, task_ids=None):
        """
        This adds a list of samples to the system. TM

        Args:
            z_list (torch.Tensor): list of samples to add
            labels: Optional list of labels corresponding to samples
            task_ids: Optional list of task IDs corresponding to samples
        """
        dsave("clm add_multi triggered")

        self.fit_reducer(z_list)
        if labels is not None and task_ids is not None:
            [
                self.add(z, label, task_id)
                for z, label, task_id in zip(z_list, labels, task_ids)
            ]
        elif labels is not None:
            [self.add(z, label) for z, label in zip(z_list, labels)]
        elif task_ids is not None:
            [self.add(z, task_id=task_id) for z, task_id in zip(z_list, task_ids)]
        else:
            [self.add(z) for z in z_list]

    def get_clusters_for_training(self):
        """Gets the samples currently stored in the clusters

        Returns:
            torch.Tensor: an array of samples currently stored in the clusters
        """
        dsave("clm get_clusters_for_training triggered")

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
        dsave("clm get_clusters_with_labels triggered")

        all_samples = []
        all_labels = []
        for cluster in self.clusters:
            for sample, label in zip(cluster.samples, cluster.labels):
                all_samples.append(sample)
                all_labels.append(label)
        samples_array = torch.stack(all_samples) if all_samples else torch.tensor([])
        return samples_array, all_labels

    def visualize(self, subtitle=""):
        """
        Show what's stored inside!
        """
        dsave("clm visualize triggered")
        # Show each cluster in it's own color
        import plotly.express as px
        from sklearn.decomposition import PCA

        if not self.clusters:
            print("No clusters to visualize")
            return

        # Collect all samples with cluster labels, task IDs, and insertion order
        all_samples = []
        cluster_labels = []
        task_labels = []
        age_labels = []

        for cluster_idx, cluster in enumerate(self.clusters):
            for sample, task_id, insertion_order in zip(
                cluster.samples, cluster.task_ids, cluster.insertion_order
            ):
                all_samples.append(sample)
                cluster_labels.append(f"Cluster {cluster_idx}")
                task_labels.append(
                    f"Task {task_id + 1}" if task_id is not None else "Unknown"
                )
                # Invert insertion_order so oldest samples (lower insertion_order) are biggest
                age_labels.append(
                    max(cluster.insertion_order) - insertion_order + 1
                    if insertion_order is not None
                    else 1
                )

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
                "Task": task_labels,
                "Age": age_labels,
            }
        )

        # Create 3D scatter plot
        fig = px.scatter_3d(
            df,
            x=x_col,
            y=y_col,
            z=z_col,
            color="Cluster",
            symbol="Task",
            size="Age",
            title="Cluster Visualization (Color=Cluster, Symbol=Task, Size=Age)",
            subtitle=subtitle,
            hover_data={"Cluster": True, "Task": True, "Age": True},
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
            torch.tensor([45, 45, 45], dtype=torch.float32) + torch.randn(3) * 2
        )
        # Cluster 2: around [50, 50, 50]
        samples.append(
            torch.tensor([50, 50, 50], dtype=torch.float32) + torch.randn(3) * 2
        )
        # Cluster 3: around [90, 90, 90]
        samples.append(
            torch.tensor([90, 90, 90], dtype=torch.float32) + torch.randn(3) * 2
        )
        # Cluster 4: Around [85, 85, 85]
        samples.append(
            torch.tensor([90, 90, 90], dtype=torch.float32) + torch.randn(3) * 2
        )
    # Outliers
    for i in range(NUM_SAMPLES - (samples_per_cluster * 3)):
        samples.append(
            torch.tensor([25 * (i + 1), 75, 25 * (i + 1)], dtype=torch.float32)
            + torch.randn(3) * 3
        )
    # assert NUM_SAMPLES == len(samples), "Did not get the expected number of samples"

    # Shuffle samples
    # random.shuffle(samples)
    # samples = reversed(samples)

    print(storage.get_clusters_with_labels())
    for i, sample in enumerate(samples):
        storage.add(sample, label=i)
        print(storage.get_clusters_with_labels())

        if VISUALIZE:
            storage.visualize()
            time.sleep(5)

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
        + ["purple"] * samples_per_cluster
        + ["black"] * (NUM_SAMPLES - samples_per_cluster)
    )
    # random.shuffle(colors)

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
    VISUALIZE = True
    if VISUALIZE:
        fig.show()
