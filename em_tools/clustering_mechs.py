# Handles the clusters themselves - assigning and removing as needed
import torch
import pandas
from collections import deque

DEBUG = False
EPSILON = 0
DEFAULT = True

triggered = set()
dprint = lambda s: triggered.add(s) if DEBUG else None


# Distance heuristic
def distance_h(*args, **kwargs):
    ans = torch.linalg.vector_norm(*args, **kwargs)
    assert ans >= 0
    return ans


class Buffer:
    def __init__(self, distance, sample, label=None, task_id=None):
        self.sample = sample
        self.label = label
        self.task_id = task_id
        self.distance = distance


counter = {}
dcount = lambda s: counter.update({s: counter.get(s, 0) + 1})


def get_final_count():
    return counter


class Cluster:
    """Represents a single cluster in the clustering mechanism."""

    def __init__(
        self, initial_sample: torch.Tensor, initial_label=None, initial_task_id=None
    ):
        dprint("cl init triggered")

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

        # Testing
        if not DEFAULT: 
            self.new_buffer: Buffer | None = None

    def __len__(self):
        if DEFAULT:
            return len(self.samples)
        return len(self.samples) + (1 if self.new_buffer is not None else 0)

    def get_sample_count(self):
        """Returns number of actual samples (excluding buffer)."""
        return len(self.samples)

    def get_sample_at_index(self, idx):
        """Returns (sample, label) at given index in samples (excluding buffer)."""
        if idx >= len(self.samples):
            raise IndexError("Sample index out of range")
        return self.samples[idx], self.labels[idx]

    def add_sample(self, sample: torch.Tensor, label=None, task_id=None):
        """Adds a new sample to the cluster and updates its mean."""
        dprint("cl add_sample triggered")
        if len(self) < 2 or DEFAULT:
            return self._add_sample(sample, label, task_id)

        # Check distance between mean and each sample, find max
        curr_distances = [distance_h(self.mean - s) for s in self.samples]
        max_curr_dist = max(curr_distances)
        dist_to_new = distance_h(self.mean - sample)

        # print(dist_to_new, max_curr_dist, abs(max_curr_dist - dist_to_new))

        if dist_to_new <= max_curr_dist + EPSILON:
            # If the new sample is closer to the mean than any current sample
            # with some variance, add it to the cluster
            dcount("save new sample")
            if (
                self.new_buffer is not None
                and distance_h(self.new_buffer.sample - sample)
                <= max_curr_dist + EPSILON
            ):
                # If the buffered sample is closer to the new sample than the
                # maximum distance between any current sample and the mean, add it as well
                dcount("added buffer because of new sample")
                self._add_sample(self.new_buffer)
            self.new_buffer = None
            return self._add_sample(sample, label, task_id)
        elif (
            self.new_buffer is not None
            and dist_to_new <= self.new_buffer.distance + EPSILON
        ):
            # If the buffer is further from the center than the new sample is,
            # Add it and the new sample. (NOTE: Logic a bit whacky?)
            dcount("save because of buffered sample")
            self._add_sample(self.new_buffer)
            self.new_buffer = None
            return self._add_sample(sample, label, task_id)
        else:
            dcount("buffer new sample")
            if self.new_buffer is not None:
                dcount("buffer overwritten before adding")
            self.new_buffer = Buffer(dist_to_new, sample, label, task_id)

    def _add_sample(self, sample, label=None, task_id=None):
        if not DEFAULT and isinstance(sample, Buffer):
            self.samples.append(sample.sample)
            self.labels.append(sample.label)
            self.task_ids.append(sample.task_id)
            # Overwrite the buffer obj with the sample itself for mean calculation
            sample = sample.sample
        else:
            self.samples.append(sample)
            self.labels.append(label)
            self.task_ids.append(task_id)
        assert isinstance(self.samples[-1], torch.Tensor), "Did not add a tensor!"
        self.insertion_order.append(self.next_insertion_id)
        self.next_insertion_id += 1
        self.sum_samples += sample.clone().detach()
        self.mean = self.sum_samples / len(self.samples)

    def remove_one(self):
        """
        Removes a sample from the cluster and updates its mean.
        """
        dprint("cl remove_one triggered")

        return self.remove_oldest()

    def remove_based_on_mean(self):
        """
        Removes the sample furthest from the mean, excluding the newest sample.
        """
        dprint("cl remove_based_on_mean triggered")

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
        dprint("cl remove_oldest triggered")

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
        dprint("clm init triggered")

        self.clusters: list[Cluster] = []  # List of Cluster objects
        self.Q = Q  # Max number of clusters
        self.P = P  # Max cluster size
        self.dimensionality_reducer = dimensionality_reducer

    def __len__(self):
        return sum([len(c) for c in self.clusters])

    def add(self, z: torch.Tensor, label=None, task_id=None):
        """
        Adds a sample z to the appropriate cluster or forms a new one.

        Args:
            z (torch.Tensor): The sample (e.g., activation vector) to add.
            label: Optional label associated with the sample (does not affect clustering).
            task_id: Optional task ID to track which task this sample came from.
        """
        dprint("clm add triggered")

        assert len(z.shape) == 1, "Sample should only have one axis."

        # Apply dimensionality reduction if configured
        if self.dimensionality_reducer is not None:
            if not self.dimensionality_reducer.fitted:
                raise ValueError(
                    "Dimensionality reducer must be fitted before adding samples. Call fit_reducer() first."
                )
            z = self.dimensionality_reducer.transform(z)
        # if z is a set of samples, add it one by one
        if len(self.clusters) < self.Q:
            # If the number of clusters is less than Q, create a new cluster
            # and add z to it.
            new_cluster = Cluster(z, label, task_id)
            self.clusters.append(new_cluster)
        else:
            # If the number of clusters has reached Q, find the closest cluster
            # based on Euclidean distance to its mean.
            min_distance = float("inf")
            closest_cluster_idx = -1

            for i, cluster in enumerate(self.clusters):
                # Calculate distance
                distance = distance_h(z - cluster.mean)
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster_idx = i

            # Add z to the identified closest cluster
            q_star = self.clusters[closest_cluster_idx]
            q_star.add_sample(z, label, task_id)

            # If the cluster size exceeds P, remove a sample
            while len(q_star.samples) > self.P:
                q_star.remove_one()

    def fit_reducer(self, z_list):
        """
        Fit dimensionality reducer on a set of samples. Must be called before adding samples if reducer is configured.

        Args:
            z_list (list or torch.Tensor): List of samples to fit reducer on
        """
        dprint("clm fit_reducer triggered")

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
        dprint("clm transform triggered")

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
        dprint("clm add_multi triggered")

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
        dprint("clm get_clusters_for_training triggered")

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
        dprint("clm get_clusters_with_labels triggered")

        all_samples = []
        all_labels = []
        for cluster in self.clusters:
            for sample, label in zip(cluster.samples, cluster.labels):
                all_samples.append(sample)
                all_labels.append(label)
        samples_array = torch.stack(all_samples) if all_samples else torch.tensor([])
        return samples_array, all_labels

    def get_total_sample_count(self):
        """Returns total number of actual samples across all clusters (excluding buffers)."""
        return sum(cluster.get_sample_count() for cluster in self.clusters)

    def get_sample_at_global_index(self, global_idx):
        """Returns (sample, label) at global index across all clusters (excluding buffers)."""
        current_offset = 0
        for cluster in self.clusters:
            cluster_size = cluster.get_sample_count()
            if global_idx < current_offset + cluster_size:
                local_idx = global_idx - current_offset
                return cluster.get_sample_at_index(local_idx)
            current_offset += cluster_size
        raise IndexError("Global sample index out of range")

    def visualize(self):
        """
        Show what's stored inside!
        """
        dprint("clm visualize triggered")
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

    print(get_final_count())
