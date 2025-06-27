import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ta-clustering'))

from clustering_gemini import ClusteringMechanism
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
        self.clustering_mechanism = ClusteringMechanism(Q=Q, P=P)
        self.input_type = input_type

    def get_memory_samples(self):
        """Returns all samples currently stored.

        Returns:
            np.ndarray: Array of all stored samples
        """
        samples, _ = self.clustering_mechanism.get_clusters_with_labels()
        return samples

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
