import torch
from clustering_gemini import ClusteringMechanism


class ClusteringMemory():
    def __init__(self, Q, P, input_type, device, num_pools=10):
        """
        Initialize multi-pool clustering memory wrapper.

        Args:
            Q (int): Maximum number of clusters per pool
            P (int): Maximum samples per cluster
            input_type (str): Type of input ('samples' for TA-A-GEM)
            num_pools (int): Number of separate pools (10 for permutation/rotation, 2 for class split)
            device: PyTorch device to use for tensor operations
        """
        self.device = device
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
            # Create new pool for this label - pass device to ClusteringMechanism
            self.pools[label] = ClusteringMechanism(Q=self.Q, P=self.P, device=self.device)
            self.pool_labels.add(label)

        return self.pools[label]

    def get_memory_samples(self):
        """Returns all samples currently stored across all pools as tuples of (sample, label)."""
        all_samples = []
        for pool in self.pools.values():
            samples, labels = pool.get_clusters_with_labels()
            # samples and labels should already be tensors from ClusteringMechanism
            for s, l in zip(samples, labels):
                # Ensure tensors are on correct device
                if isinstance(s, torch.Tensor):
                    s = s.to(self.device)
                else:
                    s = torch.tensor(s, dtype=torch.float32, device=self.device)

                if isinstance(l, torch.Tensor):
                    l = l.to(self.device)
                else:
                    l = torch.tensor(l, dtype=torch.long, device=self.device)

                all_samples.append((s, l))
        return all_samples

    def add_sample(self, sample_data, sample_label):
        """Add a sample to the appropriate pool based on its label.

        Args:
            sample_data: The sample to add (should be a tensor)
            sample_label: Label associated with the sample (determines which pool)
        """
        # Ensure sample_data is a tensor on the correct device
        if not isinstance(sample_data, torch.Tensor):
            sample_data = torch.tensor(sample_data, dtype=torch.float32, device=self.device)
        else:
            sample_data = sample_data.to(self.device)

        # Flatten if needed for MNIST
        if len(sample_data.shape) > 1:
            sample_data = sample_data.flatten()

        # Convert label to int if tensor
        if isinstance(sample_label, torch.Tensor):
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