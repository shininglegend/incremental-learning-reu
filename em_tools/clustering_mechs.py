# Handles the clusters themselves - assigning and removing as needed
import math
import random
import torch
import pandas
from collections import deque

DIST_THRESHOLD_OVERRIDE = 1.5

DEBUG = False

triggered = set()
dtriggered = lambda s: triggered.add(s) if DEBUG else None
# Helper function that allows conditional function execution based on flag above
debug = lambda f, *args, **kwargs: f(*args, **kwargs) if DEBUG else None

counter = {}
dcount = lambda s: counter.update({s: counter.get(s, 0) + 1})


def get_final_count():
    return counter


# Distance heuristic
def distance_h(*args, **kwargs):
    return torch.linalg.vector_norm(*args, **kwargs)


class Cluster:
    """Represents a single cluster in the clustering mechanism."""

    def __init__(
        self,
        initial_sample: torch.Tensor,
        initial_label=None,
        initial_task_id=None,
        random_add_or_remove=False,
    ):
        dtriggered("cl init triggered")

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
        self.random_add_or_remove = random_add_or_remove

        # Sanity check
        assert self.random_add_or_remove in [True, False]

    def __len__(self):
        return len(self.samples)

    def __str__(self):
        return f"Cluster with mean {self.mean} and labeled samples: " + "\n".join(
            [
                (f"({sample}, {label})")
                for sample, label in zip(self.samples, self.labels)
            ]
        )

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

    def add_sample(self, sample: torch.Tensor, label=None, task_id=None):
        """Adds a new sample to the cluster and updates its mean."""
        dtriggered("cl add_sample triggered")

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
        dtriggered("cl remove_one triggered")
        if self.random_add_or_remove:
            # Remove any sample (including most recently added)
            sample_idx = random.randint(0, len(self.samples) - 1)
            self._remove_sample(sample_idx)
        else:
            return self.remove_oldest_with_distance_override()

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
            distance = distance_h(sample - self.mean)
            if distance > max_distance:
                max_distance = distance
                furthest_idx = i

        # Remove the furthest sample and its metadata
        return self._remove_sample(furthest_idx)

    def remove_oldest(self):
        """
        Removes a sample from the cluster and updates its mean.
        Currently removes the oldest sample.
        """
        dtriggered("cl remove_oldest triggered")

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

    def get_oldest_task_id(self):
        """
        Returns the oldest task ID in this cluster.

        Returns:
            int or None: The oldest task ID, or None if cluster is empty or has no task IDs
        """
        if len(self.task_ids) == 0 or self.task_ids[0] is None:
            return None
        return min(self.task_ids)

    def remove_oldest_with_distance_override(self):
        """
        Removes oldest sample (leftmost in deque) unless a newer sample is much worse.
        New sample must be 1.5x worse than old sample to get removed.
        {'removed newest': 21716, 'removed oldest': 97986}
        """
        if len(self.samples) < 2:
            dcount("removed oldest, not enough samples")
            return self.remove_oldest()

        indx_of_new = -1
        dist_to_new = distance_h(self.samples[-1] - self.mean)

        # Remove the bad new sample if it's significantly worse (DIST_THRESHOLD_OVERRIDE) further
        if dist_to_new > (
            distance_h(self.samples[0] - self.mean) * DIST_THRESHOLD_OVERRIDE
        ):
            dcount("removed newest")
            return self._remove_sample(indx_of_new)
        else:
            dcount("removed oldest")
            return self.remove_oldest()

    def remove_oldest_dist_override_v2(self):
        """
        Removes oldest sample (leftmost in deque) unless one of the other two samples is worse.
        Newer samples must be 1.5x worse than the old sample to get removed.
        {'remove oldest': 112091, 'remove newest': 7611}
        """
        if len(self.samples) < 3:
            dcount("not enough samples, removed oldest")
            return self.remove_oldest()

        # Oldest is at index 0, newest at index -1
        oldest_distance = distance_h(self.samples[0] - self.mean)

        # Find worst distance among newer samples (indices 1 to n - 1)
        sample_w_biggest_dist = -1
        biggest_dist_found = -math.inf

        for i in range(1, len(self.samples) - 1):  # Checks two middle samples
            distance = distance_h(self.samples[i] - self.mean)
            if distance > biggest_dist_found:
                biggest_dist_found = distance
                sample_w_biggest_dist = i

        # Remove the oldest sample only if all other samples except for the newest are somewhat closer
        if biggest_dist_found > (oldest_distance * DIST_THRESHOLD_OVERRIDE):
            dcount("remove newest")
            return self._remove_sample(sample_w_biggest_dist)
        else:
            dcount("remove oldest")
            return self.remove_oldest()

    def _remove_sample(self, idx):
        """
        Removes the sample (and its metadata) at `idx`.
        Keeps deques intact, maintains insertion_order, mean, and sum_samples.
        """

        # Fast aliases
        S, L, T, O = self.samples, self.labels, self.task_ids, self.insertion_order

        # Grab the items we’ll return before they’re gone
        removed_sample = S[idx]
        removed_label = L[idx]

        # Bring target to the left end
        S.rotate(-idx)
        L.rotate(-idx)
        T.rotate(-idx)
        O.rotate(-idx)

        # Pop from the left
        S.popleft()
        L.popleft()
        T.popleft()
        O.popleft()

        # Restore original order
        S.rotate(idx)
        L.rotate(idx)
        T.rotate(idx)
        O.rotate(idx)

        self.sum_samples -= removed_sample
        if len(self.samples) > 0:
            self.mean = self.sum_samples / len(self.samples)
        else:
            self.mean = torch.zeros_like(self.mean)

        return removed_sample, removed_label


class ClusterPool:
    """Implements the clustering mechanism described in Algorithm 3."""

    def __init__(
        self, Q=100, P=3, add_remove_randomly=False, dimensionality_reducer=None
    ):
        """
        Initializes the clustering mechanism.

        Args:
            Q (int): Maximum number of clusters.
            P (int): Maximum size of each cluster.
            dimensionality_reducer (DimensionalityReducer): Dimensionality reduction method. If None, no reduction is applied.
        """
        dtriggered("clm init triggered")

        self.clusters: list[Cluster] = []  # List of Cluster objects
        self.max_clusters = Q  # Max number of clusters
        self.max_cluster_size = P  # Max cluster size
        self.dimensionality_reducer = dimensionality_reducer
        self.sample_throughput = 0
        self.add_rem_rand = add_remove_randomly

        assert self.max_cluster_size > 0
        assert self.max_clusters > 0

    def __len__(self):
        return sum([len(cluster) for cluster in self.clusters])

    def add(self, z: torch.Tensor, label=None, task_id=None):
        """
        Adds a sample z to the appropriate cluster or forms a new one.

        Args:
            z (torch.Tensor): The sample (e.g., activation vector) to add.
            label: Optional label associated with the sample (does not affect clustering).
            task_id: Optional task ID to track which task this sample came from.
        """
        dtriggered("clm add triggered")
        self.sample_throughput += 1

        assert len(z.shape) == 1, "Sample should only have one axis."

        # Apply dimensionality reduction if configured
        if self.dimensionality_reducer is not None:
            if not self.dimensionality_reducer.fitted:
                raise ValueError(
                    "Dimensionality reducer must be fitted before adding samples. Call fit_reducer() first."
                )
            z = self.dimensionality_reducer.transform(z)

        # If add randomly, add to a random cluster
        if self.add_rem_rand:
            if len(self.clusters) < self.max_clusters:
                return self._add_new_cluster(
                    z=z,
                    add_or_remove_randomly=self.add_rem_rand,
                    label=label,
                    task_id=task_id,
                )
            return self._add_to_cluster(
                random.choice(range(len(self.clusters))), z, label, task_id
            )

        # If we're at 0 samples, or at 1 sample and there's space for a second one, add it
        if len(self.clusters) == 0 or (
            len(self.clusters) == 1 and self.max_clusters > 1
        ):
            return self._add_new_cluster(
                z=z,
                add_or_remove_randomly=self.add_rem_rand,
                label=label,
                task_id=task_id,
            )

        # Find the closest cluster
        (nearest_cluster_idx_to_new, dist_to_new) = self._find_closest_cluster(z)

        # Check the size of it - if bigger than one, just add
        # This prevents an outlier from breaking things too much
        if len(self.clusters[nearest_cluster_idx_to_new]) > 1:
            return self._add_to_cluster(nearest_cluster_idx_to_new, z, label, task_id)

        # Otherwise, grab that cluster, check it's nearest distance.
        # If that cluster is closer to another cluster than this sample is to it, move it to that cluster
        # and remove the old cluster, replacing it with a new sample centered on this new sample
        (nearest_old_sample, nearest_old_label) = self.clusters[
            nearest_cluster_idx_to_new
        ].get_sample_or_samples()
        (closest_cluster_to_old, dist_to_old) = self._find_closest_cluster(
            nearest_old_sample, ignore=nearest_cluster_idx_to_new
        )
        if dist_to_new > dist_to_old:
            debug(self.visualize, "Before")
            # Overwrite it with the new sample
            self._add_to_cluster(
                closest_cluster_to_old, nearest_old_sample, nearest_old_label, task_id
            )
            self.clusters[nearest_cluster_idx_to_new] = Cluster(
                initial_sample=z,
                random_add_or_remove=self.add_rem_rand,
                initial_label=label,
                initial_task_id=task_id,
            )
            debug(self.visualize, "After overwriting")
            debug(print, f"Overwrote! {dist_to_new} > {dist_to_old}")
        else:
            if len(self.clusters) < self.max_clusters:
                # Add a new cluster if possible
                self._add_new_cluster(
                    z=z,
                    add_or_remove_randomly=self.add_rem_rand,
                    label=label,
                    task_id=task_id,
                )
            else:
                # Add to whatever cluster is closest
                self._add_to_cluster(nearest_cluster_idx_to_new, z, label, task_id)
            # self.visualize("After keeping")
            debug(print, f"Kept! {dist_to_new} <= {dist_to_old}")

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

    def _add_new_cluster(
        self, z, add_or_remove_randomly=False, label=None, task_id=None
    ):
        """This adds a new cluster with the given sample and label

        Args:
            z (Tensor): Sample to be added
            label (, optional): Label to assign to the sample. Defaults to None.
        """
        new_cluster = Cluster(
            initial_sample=z,
            random_add_or_remove=add_or_remove_randomly,
            initial_label=label,
            initial_task_id=task_id,
        )
        self.clusters.append(new_cluster)

    def _add_to_cluster(self, cluster_index, z, label=None, task_id=None):
        """Adds a sample to a given cluster, removing one from that cluster if needed to make space

        Args:
            cluster_index (int): Index of the cluster
            z (Tensor): Sample to add
            label (_type_, optional): Optional label of sample. Defaults to None.
        """
        # Add z to the identified closest cluster
        q_star: Cluster = self.clusters[cluster_index]
        q_star.add_sample(z, label, task_id)

        # If the cluster size exceeds P, remove a sample
        if len(q_star.samples) > self.max_cluster_size:
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

    def add_multi(self, z_list, labels=None, task_ids=None):
        """
        This adds a list of samples to the system. TM

        Args:
            z_list (torch.Tensor): list of samples to add
            labels: Optional list of labels corresponding to samples
            task_ids: Optional list of task IDs corresponding to samples
        """
        dtriggered("clm add_multi triggered")

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

    def get_oldest_task_ids(self):
        """Gets the oldest task ID from each cluster.

        Returns:
            list: List of oldest task IDs for each cluster, padded with None for unused cluster slots
        """
        oldest_task_ids = [
            cluster.get_oldest_task_id() if cluster else None
            for cluster in self.clusters
        ]

        # Pad with None for unused cluster slots up to max_clusters
        while len(oldest_task_ids) < self.max_clusters:
            oldest_task_ids.append(None)

        return oldest_task_ids

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
        title = "Cluster Visualization (Color=Cluster, Symbol=Label)"
        title += " | " + title_postfix if title_postfix != "" else ""
        fig = px.scatter_3d(
            df,
            x=x_col,
            y=y_col,
            z=z_col,
            color="Cluster",
            symbol="Task",
            title=title,
            hover_data={"Cluster": True, "Task": True},
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

    storage = ClusterPool(Q=CLUSTERS, P=MAX_PER_CLUSTER)
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
