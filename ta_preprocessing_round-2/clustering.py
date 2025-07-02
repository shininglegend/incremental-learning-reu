import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ta-clustering'))

from clustering_gemini import ClusteringMechanism
import numpy as np

class ClusteringMemory():
    def __init__(self, Q, P, input_type, num_pools=10):
        """
        Initialize multi-pool clustering memory wrapper.

        Args:
            Q (int): Maximum number of clusters per pool
            P (int): Maximum samples per cluster
            input_type (str): Type of input ('samples' for TA-A-GEM)
            num_pools (int): Number of separate pools (10 for permutation/rotation, 2 for class split)
        """
        self.Q = Q
        self.P = P
        self.input_type = input_type
        self.num_pools = num_pools

        # Create separate clustering mechanisms for each pool (class label)
        self.pools = {}
        self.pool_labels = set()  # Track which labels we've seen

    def _get_or_create_pool(self, label):
        """Get clustering mechanism for a label, creating if needed.

        Args:
            label (int): Class label to get pool for

        Returns:
            ClusteringMechanism: The clustering mechanism for this label
        """
        if label not in self.pools:
            # Create new pool for this label
            self.pools[label] = ClusteringMechanism(Q=self.Q, P=self.P, dimensionality_reducer=None)
            self.pool_labels.add(label)

        return self.pools[label]

    def get_memory_samples(self):
        """Returns all samples currently stored across all pools as tuples of (sample, label).

        Returns:
            list: List of (sample, label) tuples from all pools
        """
        all_memory_samples = []

        # Collect samples from all pools
        for label, pool in self.pools.items():
            samples, labels = pool.get_clusters_with_labels()
            if len(samples) == 0:
                continue

            # Convert numpy arrays back to tensors and pair with labels
            import torch
            for sample, sample_label in zip(samples, labels):
                sample_tensor = torch.FloatTensor(sample)
                label_tensor = torch.tensor(sample_label) if sample_label is not None else torch.tensor(label)
                all_memory_samples.append((sample_tensor, label_tensor))

        return all_memory_samples

    def add_sample(self, sample_data, sample_label):
        """Add a sample to the appropriate pool based on its label.

        Args:
            sample_data: The sample to add (tensor or array)
            sample_label: Label associated with the sample (determines which pool)
        """
        # Convert tensor to numpy if needed
        if hasattr(sample_data, 'detach'):
            sample_data = sample_data.detach().cpu().numpy()

        # Flatten if needed for MNIST
        if len(sample_data.shape) > 1:
            sample_data = sample_data.flatten()

        # Convert label to int if tensor
        if hasattr(sample_label, 'item'):
            sample_label = sample_label.item()

        # Get the appropriate pool for this label and add the sample
        pool = self._get_or_create_pool(sample_label)
        pool.add(sample_data, sample_label)

    def get_memory_size(self):
        """Get the current number of samples stored across all pools.

        Returns:
            int: Total number of samples currently stored across all pools
        """
        total_samples = 0
        for pool in self.pools.values():
            samples, _ = pool.get_clusters_with_labels()
            total_samples += len(samples)
        return total_samples

    def get_pool_sizes(self):
        """Get the number of samples in each pool.

        Returns:
            dict: Mapping from label to number of samples in that pool
        """
        pool_sizes = {}
        for label, pool in self.pools.items():
            samples, _ = pool.get_clusters_with_labels()
            pool_sizes[label] = len(samples)
        return pool_sizes

    def get_clustering_mechanism(self):
        """Get access to all clustering mechanisms for visualization.

        Returns:
            dict: Mapping from label to ClusteringMechanism instance
        """
        return self.pools

    def get_num_active_pools(self):
        """Get the number of pools that have been created.

        Returns:
            int: Number of active pools
        """
        return len(self.pools)
