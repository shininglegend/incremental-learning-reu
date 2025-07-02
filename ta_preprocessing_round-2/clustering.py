from ta_clustering.clustering_gemini import ClusteringMechanism
import numpy as np


class ClusteringMemory():
    def __init__(self, Q, P, input_type):
        """
        Initialize clustering memory wrapper.

        Args:
            Q (int): Maximum number of clusters
            P (int): Maximum samples per cluster
            input_type (str): Type of input ('samples' for TA-A-GEM)
        """
        self.clustering_mechanism = ClusteringMechanism(Q=Q, P=P, dimensionality_reducer=None)
        self.input_type = input_type

    def get_memory_samples(self):
        """Returns all samples currently stored as tuples of (sample, label).

        Returns:
            list: List of (sample, label) tuples
        """
        samples, labels = self.clustering_mechanism.get_clusters_with_labels()
        if len(samples) == 0:
            return []

        # Convert numpy arrays back to tensors and pair with labels
        import torch
        memory_samples = []
        for sample, label in zip(samples, labels):
            sample_tensor = torch.FloatTensor(sample)
            label_tensor = torch.tensor(label) if label is not None else torch.tensor(0)
            memory_samples.append((sample_tensor, label_tensor))

        return memory_samples

    def add_sample(self, sample_data, sample_label):
        """Add a sample to the clustering memory.

        Args:
            sample_data: The sample to add (tensor or array)
            sample_label: Label associated with the sample
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

        self.clustering_mechanism.add(sample_data, sample_label)

    def get_memory_size(self):
        """Get the current number of samples stored in memory.

        Returns:
            int: Number of samples currently stored
        """
        samples, _ = self.clustering_mechanism.get_clusters_with_labels()
        return len(samples)

    def get_clustering_mechanism(self):
        """Get access to the underlying clustering mechanism for visualization.

        Returns:
            ClusteringMechanism: The clustering mechanism instance
        """
        return self.clustering_mechanism
