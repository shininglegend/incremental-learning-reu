# Handles the pools of clusters
try:
    from clustering_mechs import ClusterPool
except ImportError:
    from em_tools.clustering_mechs import ClusterPool

import torch


class ClusteringMemory:
    def __init__(self, Q, P, input_type, device, config, num_pools=10):
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
        self.total_samples = 0
        self.max_samples = self.Q * self.P * self.num_pools

        # Create separate clustering mechanisms for each pool (class label)
        self.pools = {}
        self.pool_labels = set()  # Track which labels we've seen

        # For initializing clusters
        self.cluster_params = {
            'removal': config['removal'],
            'consider_newest': config['consider_newest'],
            'max_size_per_cluster': self.P
        }

    def _get_or_create_pool(self, label) -> ClusterPool:
        """Get clustering mechanism for a label, creating if needed.

        Args:
            label (int): Class label to get pool for

        Returns:
            ClusteringMechanism: The clustering mechanism for this label
        """
        if label not in self.pools:
            # Create new pool for this label
            self.pools[label] = ClusterPool(
                cluster_params=self.cluster_params, Q=self.Q, P=self.P, dimensionality_reducer=None
            )
            self.pool_labels.add(label)

        return self.pools[label]

    def get_memory_samples(self, timer=None):
        """Returns all samples currently stored across all pools as tuples of (sample, label).

        Optimized version that batches tensor operations.

        Returns:
            tuple: (torch.Tensor of samples, torch.Tensor of labels) or None if no samples
        """
        ts = lambda k: timer.start(k) if timer is not None else None
        te = lambda k: timer.end(k) if timer is not None else None

        ts("total")
        all_samples = []
        all_labels = []

        # Collect samples from all pools
        for label, pool in self.pools.items():
            ts("from pool")
            samples, labels = pool.get_clusters_with_labels()
            te("from pool")

            if len(samples) == 0:
                continue

            # Add samples and labels to lists (no conversion yet)
            all_samples.append(samples)
            all_labels.extend(labels if labels else [label] * len(samples))

        if not all_samples:
            te("total")
            return None

        # Single batch conversion to device
        ts("convert")
        # Concatenate all samples at once
        all_samples_tensor = torch.cat(all_samples, dim=0).to(self.device)
        # Convert labels to tensor in one operation
        all_labels_tensor = torch.tensor(all_labels, dtype=torch.long).to(self.device)
        te("convert")

        te("total")
        return all_samples_tensor, all_labels_tensor

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
        removed = pool.add(sample_tensor, sample_label, task_id)

        # Only iterate total samples if we didn't also remove a sample from memory
        if not removed:
            self.total_samples += 1

    def get_memory_size(self):
        """Get the current number of samples stored across all pools.

        Returns:
            int: Total number of samples currently stored across all pools
        """

        return self.total_samples

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
            dict: Mapping from label to ClusterPool instance
        """
        return self.pools

    def get_num_active_pools(self):
        """Get the number of pools that have been created.

        Returns:
            int: Number of active pools
        """
        return len(self.pools)
