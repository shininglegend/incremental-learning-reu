# Handles the pools of clusters
try:
    from clustering_mechs import ClusteringMechanism
except ImportError:
    from em_tools.clustering_mechs import ClusteringMechanism

import torch
import math
import random


class ClusteringMemory:
    def __init__(self, Q, P, input_type, device, num_pools=10):
        """
        Initialize multi-pool clustering memory wrapper.

        Args:
            Q (int): Maximum number of clusters per pool
            P (int): Maximum samples per cluster
            input_type (str): Type of input ('samples' for TA-A-GEM)
            num_pools (int): Number of separate pools (10 for permutation/rotation, 2 for class split)
            device: PyTorch device to use for tensor creation
        """
        self.device = device
        self.Q = Q
        self.P = P
        self.input_type = input_type
        self.num_pools = num_pools

        # Create separate clustering mechanisms for each pool (class label)
        self.pools = {}
        self.pool_labels = set()  # Track which labels we've seen

    def _get_or_create_pool(self, label) -> ClusteringMechanism:
        """Get clustering mechanism for a label, creating if needed.

        Args:
            label (int): Class label to get pool for

        Returns:
            ClusteringMechanism: The clustering mechanism for this label
        """
        if label not in self.pools:
            # Create new pool for this label
            self.pools[label] = ClusteringMechanism(
                Q=self.Q, P=self.P, dimensionality_reducer=None
            )
            self.pool_labels.add(label)

        return self.pools[label]

    def get_memory_samples(self, timer=None):
        """Returns all samples currently stored across all pools as tuples of (sample, label).

        Returns:
            list: List of (sample, label) tuples from all pools
        """
        ts = lambda k: timer.start(k) if timer is not None else None
        te = lambda k: timer.end(k) if timer is not None else None
        all_memory_samples = []

        # Collect samples from all pools
        ts("total")
        for label, pool in self.pools.items():
            ts("from pool")
            samples, labels = pool.get_clusters_with_labels()
            te("from pool")
            if len(samples) == 0:
                continue

            # Convert tensors to device and pair with labels
            ts("convert")
            for sample, sample_label in zip(samples, labels):
                sample_tensor = sample.to(self.device)
                label_tensor = torch.tensor(
                    sample_label if sample_label is not None else label
                ).to(self.device)
                all_memory_samples.append((sample_tensor, label_tensor))
            te("convert")
        te("total")
        return all_memory_samples

    def get_random_samples(self, amount):
        if sum([len(pool) for _, pool in self.pools.items()]) <= amount:
            return self.get_memory_samples()
        random_samples = []
        get_from_each = 1
        # Choose from random pools:
        i = 0
        if amount >= len(self.pools):
            get_from_each = math.ceil(amount / len(self.pools))
        for i in range(len(self.pools)):
            i += 1
            if len(random_samples) >= amount:
                break
            random_samples.extend(
                self._get_random_samples_from_pool(
                    i - 1,
                    min(
                        (get_from_each * i) - len(random_samples),
                        amount - len(random_samples),
                    )
                )
            )
        assert amount >= len(random_samples), "Wrong number of samples"
        if amount > len(random_samples):
            print("\nNot enough samples stored to return.")
        return random_samples

    def _get_random_samples_from_pool(self, pool_idx, amount):
        # Get pool labels as list for indexing
        pool_labels = list(self.pools.keys())
        assert pool_idx < len(pool_labels), "Invalid index!"

        label = pool_labels[pool_idx]
        pool = self.pools[label]

        # Get all samples from this pool
        samples, labels = pool.get_clusters_with_labels()
        if len(samples) == 0:
            return []

        # Generate random indices for sampling
        available_count = len(samples)
        sample_count = min(amount, available_count)
        random_indices = random.sample(range(available_count), sample_count)

        # Extract samples at random indices
        result = []
        for idx in random_indices:
            sample_tensor = samples[idx].to(self.device)
            label_tensor = torch.tensor(labels[idx] if labels[idx] is not None else label).to(self.device)
            result.append((sample_tensor, label_tensor))

        return result

    def add_sample(self, sample_data, sample_label, task_id=None):
        """Add a sample to the appropriate pool based on its label.

        Args:
            sample_data: The sample to add (tensor or array)
            sample_label: Label associated with the sample (determines which pool)
            task_id: Optional task ID to track which task this sample came from
        """

        # Convert to tensor if needed
        if hasattr(sample_data, "detach"):
            # Already a tensor, just ensure it's on CPU for storage
            sample_tensor = sample_data.detach().cpu()
        else:
            # Convert from numpy or other formats to tensor
            sample_tensor = torch.tensor(sample_data, dtype=torch.float32)

        # Flatten if needed for MNIST
        if len(sample_tensor.shape) > 1:
            sample_tensor = sample_tensor.flatten()

        # Convert label to int if tensor
        if hasattr(sample_label, "item"):
            sample_label = sample_label.item()

        # Get the appropriate pool for this label and add the sample
        pool = self._get_or_create_pool(sample_label)
        pool.add(sample_tensor, sample_label, task_id)

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
        pool_sizes = {label: len(pool) for label, pool in self.pools.items()}
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
