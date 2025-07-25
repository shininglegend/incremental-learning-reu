# Handles the clusters themselves - assigning and removing as needed
import math

import torch
import pandas
from collections import deque
import random

DEBUG = False

triggered = set()
dprint = lambda s: triggered.add(s) if DEBUG else None


# Distance heuristic
def distance_h(*args, **kwargs):
    return torch.linalg.vector_norm(*args, **kwargs)


class Cluster:
    """Represents a single cluster in the clustering mechanism."""

    def __init__(
        self, cluster_params, initial_sample: torch.Tensor,
            initial_label=None, initial_task_id=None,
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
        self.cluster_params = cluster_params


        # for validating removal method at initialization
        self.removal_methods = {
            'remove_oldest': self.remove_oldest,
            'remove_furthest_from_mean': self.remove_furthest_from_mean,
            'remove_random': self.remove_random,
            'remove_furthest_from_new': self.remove_furthest_from_new,
            'remove_based_on_mean': self.remove_based_on_mean,
            'remove_closest_to_new': self.remove_closest_to_new,
            'remove_by_weighted_mean_distance': self.remove_by_weighted_mean_distance,
            'remove_by_log_weighted_mean_distance': self.remove_by_log_weighted_mean_distance,
            'remove_oldest_with_distance_override': self.remove_oldest_with_distance_override,
        }

        if self.cluster_params['removal'] in self.removal_methods:
            self.removal_fn = self.removal_methods[cluster_params['removal']]
        else:
            print("Unknown removal method detected. Performing remove_based_on_mean.")
            self.removal_fn = self.remove_based_on_mean

        self.consider_newest = self.cluster_params['consider_newest']

        dprint(f'Removal Function in cl: {self.removal_fn}')
        dprint(f'Consider newest: {self.consider_newest}')


    def add_sample(self, sample: torch.Tensor, label=None, task_id=None):
        """Adds a new sample to the cluster and updates its mean."""
        dprint("cl add_sample triggered")

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
        dprint("cl remove_one triggered")

        return self.removal_fn()

    def remove_oldest(self):
        """
        Removes the oldest sample from the cluster and updates its mean.
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
        else:
            dprint("Remove_oldest called on an empty cluster. Try again.")
            return None, None, 0, 0

    def remove_furthest_from_mean(self):
        """
        Removes the sample furthest from the mean (including the newest sample).

        From Abby: I'm considering replacing this method with just return self.remove_by_weighted_mean_distance(0)
        because a weight of 0 indicates no consideration of age when removing.
        TODO: resolve this.
        """
        dprint("cl remove_furthest_from_mean triggered")

        if len(self.samples) <= 1:
            return self.remove_oldest()

        # Find sample furthest from mean, considering all samples
        max_distance = -1
        furthest_idx = 0

        if self.consider_newest:
            iteration_range = len(self.samples)
        else:
            iteration_range = len(self.samples) - 1

        for i in range(iteration_range):
            sample = self.samples[i]
            distance = distance_h(sample - self.mean)
            if distance > max_distance:
                max_distance = distance
                furthest_idx = i

        return self._remove_by_index(furthest_idx)

    def remove_random(self):
        """
        Removes a random sample from the cluster and updates its mean.
        """
        dprint("cl remove_random triggered")

        if len(self.samples) <= 0:
            return None, None

        # Generate random index
        if self.consider_newest:
            max_idx = len(self.samples) - 1
        else:
            if len(self.samples) == 1:
                dprint("Edge case reached: remove_random is removing a sample from a 1-element cluster.")
                return self.remove_oldest()
            max_idx = len(self.samples) - 2

        random_idx = random.randint(0, max_idx)

        return self._remove_by_index(random_idx)

    def remove_furthest_from_new(self):
        """
        Removes the sample least similar to the newest one (last in deque).
        """
        dprint("cl remove_furthest_from_new triggered")

        if len(self.samples) <= 1:
            return self.remove_oldest()

        new_sample = self.samples[-1]

        max_distance = -1
        least_similar_idx = 0

        for i in range(len(self.samples) - 1):
            distance = distance_h(self.samples[i] - new_sample)
            if distance > max_distance:
                max_distance = distance
                least_similar_idx = i

        return self._remove_by_index(least_similar_idx)

    def remove_closest_to_new(self):
        """
        Removes the sample most similar to the newest one (last in deque).
        """
        dprint("cl remove_closest_to_new triggered")

        if len(self.samples) <= 1:
            return self.remove_oldest()

        new_sample = self.samples[-1]

        min_distance = float('inf')  # Start with infinity
        most_similar_idx = 0

        for i in range(len(self.samples) - 1):
            distance = distance_h(self.samples[i] - new_sample)
            if distance < min_distance:
                min_distance = distance
                most_similar_idx = i

        return self._remove_by_index(most_similar_idx)

    def remove_based_on_mean(self):
        """
        Removes the sample furthest from the mean, excluding the newest sample.
        """
        dprint("cl remove_based_on_mean triggered")
        return self.remove_furthest_from_mean()

    def remove_by_weighted_mean_distance(self, age_weight_factor=0.02):
        """
        Removes the sample with the highest weighted distance from cluster mean.
        Older samples get lower weights (are less likely to be removed).

        Args:
            age_weight_factor: Controls how much age affects removal probability.
                              Higher values = more resistance to removing old samples.
                              0.0 = no age consideration, 1.0 = strong age bias.
        """
        dprint("cl remove_by_weighted_mean_distance triggered")

        if len(self.samples) <= 1:
            return self.remove_oldest()

        max_weighted_distance = -1
        worst_sample_idx = -1

        for i in range(len(self.samples) if self.consider_newest else len(self.samples) - 1):
            # Distance from cluster mean
            distance = distance_h(self.samples[i] - self.mean)
            age = self.insertion_order[i]

            # Age weight: newer samples (higher index) get higher weights
            # This makes them more likely to be removed
            age_weight = 1.0 + age_weight_factor * age

            weighted_distance = distance * age_weight

            if weighted_distance > max_weighted_distance:
                max_weighted_distance = weighted_distance
                worst_sample_idx = i

        return self._remove_by_index(worst_sample_idx)

    def remove_by_log_weighted_mean_distance(self, age_weight_factor=0.5):
        """
        Removes the sample with the highest weighted distance from cluster mean.
        Uses logarithmic age weighting for gentle age influence.
        """
        if len(self.samples) <= 1:
            return self.remove_oldest()

        max_weighted_distance = -1
        worst_sample_idx = -1

        for i in range(len(self.samples) if self.consider_newest else len(self.samples) - 1):
            distance = distance_h(self.samples[i] - self.mean)
            age = self.insertion_order[i]

            # Logarithmic age weight - gentle scaling
            age_weight = 1.0 + age_weight_factor * math.log(1 + age)

            weighted_distance = distance * age_weight

            if weighted_distance > max_weighted_distance:
                max_weighted_distance = weighted_distance
                worst_sample_idx = i

        return self._remove_by_index(worst_sample_idx)

    def remove_oldest_with_distance_override(self, distance_threshold_multiplier=2.0):
        """
        Removes oldest sample (leftmost in deque) unless a newer sample is much worse.
        New sample must be 1.5x worse than old sample to get removed.
        """
        if len(self.samples) <= 1:
            return self.remove_oldest()

        # Oldest is at index 0, newest at index -1
        oldest_distance = distance_h(self.samples[0] - self.mean)

        # Find worst distance among newer samples (indices 1 to end)
        worst_newer_idx = 0
        worst_newer_distance = oldest_distance

        end_range = len(self.samples) if self.consider_newest else len(self.samples) - 1
        for i in range(1, end_range):
            distance = distance_h(self.samples[i] - self.mean)
            if distance > worst_newer_distance:
                worst_newer_distance = distance
                worst_newer_idx = i

        # Remove the bad newer sample if it's significantly worse
        if worst_newer_distance > oldest_distance * distance_threshold_multiplier:
            # print("override engaged. let's hack the mainframe.")
            return self._remove_by_index(worst_newer_idx)
        else:
            return self._remove_by_index(0)  # Remove oldest (leftmost)

    def _remove_by_index(self, idx: int):
        """
        Removes the sample (and its metadata) at `idx`.
        Keeps deques intact, maintains insertion_order, mean, and sum_samples.
        """

        # Fast aliases
        S, L, T, O = self.samples, self.labels, self.task_ids, self.insertion_order

        # Grab the items we’ll return before they’re gone
        removed_sample = S[idx]
        removed_label = L[idx]
        removed_task_id = T[idx] if T else None
        removed_order_id = O[idx] if O else None

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

        # Update running stats
        self.sum_samples -= removed_sample
        self.mean = self.sum_samples / len(S) if S else torch.zeros_like(self.mean)

        return removed_sample, removed_label

    def num_samples(self):
        return len(self.samples)

    def is_full(self):
        return len(self.samples) >= self.cluster_params['max_size_per_cluster']

    def __str__(self):
        return f"""Cluster with mean {self.mean} and samples {self.samples}"""


class ClusterPool:
    """Implements the clustering mechanism described in Algorithm 3."""

    def __init__(self, cluster_params, Q=100, P=3, dimensionality_reducer=None):
        """
        Initializes the clustering mechanism.

        Args:
            Q (int): Maximum number of clusters.
            P (int): Maximum size of each cluster.
            dimensionality_reducer (DimensionalityReducer): Dimensionality reduction method. If None, no reduction is applied.
        """
        dprint("clm init triggered")

        self.clusters: list[Cluster] = []  # List of Cluster objects
        self.num_samples = 0
        self.Q = Q  # Max number of clusters
        self.P = P  # Max cluster size
        self.dimensionality_reducer = dimensionality_reducer

        self.cluster_params = cluster_params

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

        removed = False

        # Apply dimensionality reduction if configured
        if self.dimensionality_reducer is not None:
            if not self.dimensionality_reducer.fitted:
                raise ValueError(
                    "Dimensionality reducer must be fitted before adding samples. Call fit_reducer() first."
                )
            z = self.dimensionality_reducer.transform(z)
        # if z is a set of samples, add it one by one
        if len(self.clusters) < self.Q:
            # If the number of clusters is less than Q, create a new cluster and add z to it.
            new_cluster = Cluster(self.cluster_params, z, label, task_id)
            self.clusters.append(new_cluster)
            self.num_samples += 1
        else:
            # If the number of clusters has reached Q, find the closest cluster
            # Add z to the identified closest cluster
            q_star = self.get_closest_cluster(z)
            q_star.add_sample(z, label, task_id)
            self.num_samples += 1

            # If the cluster size exceeds P, remove one sample
            if len(q_star.samples) > self.P:
                q_star.remove_one()
                self.num_samples -= 1
                removed = True

        return removed

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

    def get_closest_cluster(self, z: torch.Tensor):
        # based on Euclidean distance to mean.

        min_distance = float("inf")
        closest_cluster_idx = -1

        for i, cluster in enumerate(self.clusters):
            # Calculate distance
            distance = distance_h(z - cluster.mean)
            if distance < min_distance:
                min_distance = distance
                closest_cluster_idx = i

        return self.clusters[closest_cluster_idx]

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

        if not any(cluster.samples for cluster in self.clusters):
            return torch.tensor([]), []

        all_samples = []
        all_labels = []

        for cluster in self.clusters:
            if cluster.samples:  # Only process non-empty clusters
                # Convert deque to list once, then stack
                cluster_samples_list = list(cluster.samples)
                if cluster_samples_list:
                    cluster_samples = torch.stack(cluster_samples_list)
                    all_samples.append(cluster_samples)
                    all_labels.extend(cluster.labels)

        # Concatenate all cluster samples at once
        samples_tensor = torch.cat(all_samples, dim=0) if all_samples else torch.tensor([])
        return samples_tensor, all_labels

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
    VISUALIZE = True
    if VISUALIZE:
        import time

    storage = ClusterPool(Q=2, P=3)
    samples = [
        torch.randint(0, 100, (3,), dtype=torch.float32) for _ in range(NUM_SAMPLES)
    ]
    print(storage.get_clusters_with_labels())
    for i, sample in enumerate(samples):
        storage.add(sample, label=torch.randint(1, 4, (1,)).item(), task_id=i % 3)
        if VISUALIZE:
            # storage.visualize()
            print(storage.get_clusters_with_labels())
            # time.sleep(5)

    if len(storage.clusters) == 0:
        raise Exception("No samples successfully added")
    if VISUALIZE:
        storage.visualize()

    print(storage.get_clusters_with_labels())
    print(sorted(list(triggered))) if len(triggered) > 0 else None
