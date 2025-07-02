# Initial Code by gemini. 1x2 matrix per item
# Edits made by Abigail Dodd
# with help from Claude Sonnet 4.0 and Gemini 2.5
import numpy as np
import pandas
from collections import deque
import random
from sklearn.cluster import KMeans
from ta_clustering.dim_reduction import IdentityReducer

# TODO: consider dynamic cluster sizes?
#  or maybe we can implement some kind of density measure
#  for clusters that keep getting cycled

def recluster_two_clusters(cluster1, cluster2, P):
    """
    Takes two Cluster objects and tries to re-cluster them into two different Clusters
    that are better using K-means, while respecting a maximum cluster size. This
    highly optimized implementation runs multiple KMeans initializations and
    efficiently checks constraints, only creating full Cluster objects for the
    final best valid solution.

    Args:
        cluster1 (Cluster): The first Cluster object.
        cluster2 (Cluster): The second Cluster object.
        P (int): The maximum allowed size for a cluster as defined by the ClusteringMechanism.
                 New clusters will be rejected if their size is greater than P+1.

    Returns:
        tuple[Cluster, Cluster] or tuple[None, None]: The two new Cluster objects if a
        better and valid re-clustering is found, otherwise (None, None).
    """
    # Combine all samples and their original labels from both input clusters
    all_samples = np.array(list(cluster1.samples) + list(cluster2.samples))
    all_labels = list(cluster1.labels) + list(cluster2.labels)

    num_total_samples = len(all_samples)
    if num_total_samples < 2:
        # Cannot meaningfully cluster with less than 2 points
        return None, None

    # Calculate the total inertia of the current two input clusters
    current_inertia = 0.0
    for sample in cluster1.samples:
        current_inertia += np.linalg.norm(np.array(sample) - cluster1.mean) ** 2
    for sample in cluster2.samples:
        current_inertia += np.linalg.norm(np.array(sample) - cluster2.mean) ** 2

    # Initialize variables to keep track of the best valid clustering found
    best_valid_new_inertia = float('inf')
    best_valid_new_assignments = None # Store the raw assignments, not full Cluster objects

    # Number of KMeans trials to run.
    # This value (10) is based on your feedback.
    N_KMEANS_TRIALS = 10

    for i in range(N_KMEANS_TRIALS):
        # Run KMeans for a single initialization (n_init=1).
        # Vary random_state for each trial to ensure different starting points.
        kmeans_run = KMeans(n_clusters=2, random_state=i, n_init=1, algorithm='lloyd')
        new_cluster_assignments = kmeans_run.fit_predict(all_samples)
        run_inertia = kmeans_run.inertia_

        # Optimization: If the current run's inertia is already worse than or equal to
        # the best valid one found so far, or worse than the original clusters,
        # there's no need to proceed with forming clusters for this run.
        if run_inertia >= best_valid_new_inertia or run_inertia >= current_inertia:
            continue

        # --- HIGHLY OPTIMIZED SIZE CHECK ---
        # Count sizes directly from the assignments array, avoiding Cluster object creation
        size_cluster_0 = np.sum(new_cluster_assignments == 0)
        size_cluster_1 = np.sum(new_cluster_assignments == 1)

        # Validate: Ensure both proposed clusters are non-empty
        if size_cluster_0 == 0 or size_cluster_1 == 0:
            continue # KMeans resulted in an empty cluster, not a desired outcome for 2 clusters

        # Apply the P+1 size constraint: If either new cluster exceeds the maximum allowed size,
        # this particular re-clustering is invalid and we skip it.
        if size_cluster_0 > (P + 1) or size_cluster_1 > (P + 1):
            continue

        # If we've reached this point, this KMeans run resulted in a valid re-clustering
        # (non-empty and satisfying the P+1 size constraint) and is better than previous best.
        if run_inertia < best_valid_new_inertia:
            best_valid_new_inertia = run_inertia
            best_valid_new_assignments = new_cluster_assignments # Store just the assignments

    # --- FINAL CLUSTER CREATION (ONLY IF A BEST VALID CANDIDATE WAS FOUND) ---
    if best_valid_new_assignments is None:
        return None, None # No valid and improved re-clustering found across all trials

    # Initialize lists to hold samples and labels for the final chosen clusters
    final_cluster_0_samples = []
    final_cluster_0_labels = []
    final_cluster_1_samples = []
    final_cluster_1_labels = []

    # Populate the final cluster lists using the best assignments found
    for i, assignment in enumerate(best_valid_new_assignments):
        if assignment == 0:
            final_cluster_0_samples.append(all_samples[i])
            final_cluster_0_labels.append(all_labels[i])
        else:
            final_cluster_1_samples.append(all_samples[i])
            final_cluster_1_labels.append(all_labels[i])

    # Create the final Cluster objects using your original Cluster class constructor pattern
    # First cluster
    new_cluster1 = Cluster(final_cluster_0_samples[0], final_cluster_0_labels[0])
    for i in range(1, len(final_cluster_0_samples)):
        new_cluster1.add_sample(final_cluster_0_samples[i], final_cluster_0_labels[i])

    # Second cluster
    new_cluster2 = Cluster(final_cluster_1_samples[0], final_cluster_1_labels[0])
    for i in range(1, len(final_cluster_1_samples)):
        new_cluster2.add_sample(final_cluster_1_samples[i], final_cluster_1_labels[i])

    return new_cluster1, new_cluster2


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

    # Remove a cluster and update the mean.
    # TODO: change method call to not be FIFO
    def remove_one(self):
        return self.remove_oldest()

    def remove_furthest_from_mean(self):
        # Remove the sample furthest from the mean, excluding the newest sample.
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

    def remove_closest_to_mean(self):
        # Remove the sample closest to the mean, excluding the newest sample.
        if len(self.samples) <= 1:
            return self.remove_oldest()

        # Find sample closest to mean
        min_distance = float('inf')
        closest_idx = 0

        for i in range(len(self.samples)):
            sample = self.samples[i]
            distance = np.linalg.norm(np.array(sample) - self.mean)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i

        # Remove the closest sample and its label
        removed_sample = self.samples[closest_idx]
        removed_label = self.labels[closest_idx]

        # Convert deque to list for index-based removal
        samples_list = list(self.samples)
        labels_list = list(self.labels)

        samples_list.pop(closest_idx)
        labels_list.pop(closest_idx)

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
        # Removes the oldest sample from the cluster and updates its mean.
        if len(self.samples) > 0:
            oldest_sample = self.samples.popleft()
            self.labels.popleft() # Also remove the corresponding label
            self.sum_samples -= np.array(oldest_sample)
            if len(self.samples) > 0:
                self.mean = self.sum_samples / len(self.samples)
            else:
                self.mean = np.zeros_like(self.mean) # If cluster becomes empty

    def remove_random_point(self):
        # Removes a random point from the cluster and updates accordingly
        if len(self.samples) == 0:
            return None, None

        if len(self.samples) == 1:
            return self.remove_oldest()

        # Choose random index
        random_idx = random.randint(0, len(self.samples) - 1)

        # Remove the random sample and its label
        samples_list = list(self.samples)
        labels_list = list(self.labels)

        removed_sample = samples_list.pop(random_idx)
        removed_label = labels_list.pop(random_idx)

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

    def calculate_total_distance_from_mean(self):
        # Returns the points' total distance from cluster mean
        if len(self.samples) == 0:
            return 0.0

        total_distance = 0.0
        for sample in self.samples:
            distance = np.linalg.norm(np.array(sample) - self.mean)
            total_distance += distance

        return total_distance

    def calculate_average_distance_from_mean(self):
        # Returns the points' average distance from the mean
        if len(self.samples) == 0:
            return 0.0

        total_distance = self.calculate_total_distance_from_mean()
        return total_distance / len(self.samples)

    def _update_mean(self):
        # Helper to update the cluster mean
        if self.samples:
            self.mean = np.mean(self.samples, axis=0)
        else:
            self.mean = None  # Cluster is empty

    def __str__(self):
        return f"""Cluster with mean {self.mean} and samples {self.samples}"""


class ClusteringMechanism:
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

    def add(self, z: np.ndarray, label=None):
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
                raise ValueError("Dimensionality reducer not fitted before adding samples. Call fit_reducer() first.")
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
            second_min_distance = float('inf')
            closest_cluster_idx = -1
            second_closest_cluster_idx = -1

            for i, cluster in enumerate(self.clusters):
                # Calculate Euclidean distance
                distance = np.linalg.norm(z - cluster.mean)
                if distance < min_distance:
                    second_min_distance = min_distance
                    second_closest_cluster_idx = closest_cluster_idx
                    min_distance = distance
                    closest_cluster_idx = i
                elif distance < second_min_distance:
                    second_min_distance = distance
                    second_closest_cluster_idx = i

            # Add z to the identified closest cluster
            q_star = self.clusters[closest_cluster_idx]
            q_star.add_sample(z, label)

            # If the cluster size exceeds P, check if reclustering is worth it
            if len(q_star.samples) > self.P:
                s_star = self.clusters[second_closest_cluster_idx]

                # Check if the new point z is the furthest from both cluster means
                # First check closest cluster (q_star)
                z_distance_from_q_star = np.linalg.norm(z - q_star.mean)
                is_furthest_from_q_star = True
                for sample in q_star.samples:
                    if np.array_equal(sample, z):
                        continue  # Skip the new point itself
                    sample_distance = np.linalg.norm(np.array(sample) - q_star.mean)
                    if sample_distance > z_distance_from_q_star:
                        is_furthest_from_q_star = False
                        break

                # Check second closest cluster (s_star) only if z is furthest from q_star
                is_furthest_from_s_star = False
                if is_furthest_from_q_star:
                    z_distance_from_s_star = np.linalg.norm(z - s_star.mean)
                    is_furthest_from_s_star = True
                    for sample in s_star.samples:
                        sample_distance = np.linalg.norm(np.array(sample) - s_star.mean)
                        if sample_distance > z_distance_from_s_star:
                            is_furthest_from_s_star = False
                            break

                # If z is furthest from both cluster means, skip reclustering
                if is_furthest_from_q_star and is_furthest_from_s_star:
                    # Skip reclustering and just remove one point from closest cluster
                    q_star.remove_one()
                else:
                    # Proceed with reclustering as before
                    new_cluster_one, new_cluster_two = recluster_two_clusters(q_star, s_star, self.P)
                    if new_cluster_one is None:  # None implies the first two clusters were optimal
                        q_star.remove_one()
                    else:
                        self.clusters[closest_cluster_idx] = new_cluster_one
                        self.clusters[second_closest_cluster_idx] = new_cluster_two
                        if len(new_cluster_one.samples) > self.P:
                            new_cluster_one.remove_one()
                        elif len(new_cluster_two.samples) > self.P:
                            new_cluster_two.remove_one()

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
