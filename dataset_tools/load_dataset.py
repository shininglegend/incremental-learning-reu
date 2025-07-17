"""
Abstract base class for dataset loading with domain-incremental learning support.
This module provides a common interface for loading different datasets and preparing
them for incremental learning tasks.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Any
import torch
from torch.utils.data import DataLoader


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders supporting domain-incremental learning."""

    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    @abstractmethod
    def load_raw_data(self) -> Tuple[Any, Any, Any, Any]:
        """Load raw dataset from source.

        Returns:
            tuple: (x_train, y_train, x_test, y_test) in raw format
        """
        pass

    @abstractmethod
    def preprocess_data(
        self,
        x_train: Any,
        y_train: Any,
        x_test: Any,
        y_test: Any,
        quick_test: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess raw data into PyTorch tensors.

        Args:
            x_train: Raw training images
            y_train: Raw training labels
            x_test: Raw test images
            y_test: Raw test labels
            quick_test: Whether to use reduced dataset for testing

        Returns:
            tuple: (x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor)
        """
        pass

    def prepare_domain_incremental_data(
        self, task_type: str, num_tasks: int, batch_size: int, quick_test: bool = False
    ) -> Tuple[List[DataLoader], List[DataLoader]]:
        """Prepare data for domain-incremental learning.

        Args:
            task_type: One of 'permutation', 'rotation', 'class_split'
            num_tasks: Number of tasks to create
            batch_size: Batch size for DataLoaders
            quick_test: If True, use reduced dataset for faster testing

        Returns:
            tuple: (train_dataloaders, test_dataloaders) - Lists of DataLoader objects
        """
        # Load and preprocess data
        x_train_raw, y_train_raw, x_test_raw, y_test_raw = self.load_raw_data()
        x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = (
            self.preprocess_data(
                x_train_raw, y_train_raw, x_test_raw, y_test_raw, quick_test
            )
        )

        # Store for potential future use
        self.x_train = x_train_tensor
        self.y_train = y_train_tensor
        self.x_test = x_test_tensor
        self.y_test = y_test_tensor

        # Create task-specific data loaders
        return self._create_task_dataloaders(
            x_train_tensor,
            y_train_tensor,
            x_test_tensor,
            y_test_tensor,
            task_type,
            num_tasks,
            batch_size,
        )

    def _create_task_dataloaders(
        self,
        x_train_tensor: torch.Tensor,
        y_train_tensor: torch.Tensor,
        x_test_tensor: torch.Tensor,
        y_test_tensor: torch.Tensor,
        task_type: str,
        num_tasks: int,
        batch_size: int,
    ) -> Tuple[List[DataLoader], List[DataLoader]]:
        """Create task-specific data loaders based on task type."""
        if task_type == "permutation":
            return self._create_permutation_tasks(
                x_train_tensor,
                y_train_tensor,
                x_test_tensor,
                y_test_tensor,
                num_tasks,
                batch_size,
            )
        elif task_type == "rotation":
            return self._create_rotation_tasks(
                x_train_tensor,
                y_train_tensor,
                x_test_tensor,
                y_test_tensor,
                num_tasks,
                batch_size,
            )
        elif task_type == "class_split":
            return self._create_class_split_tasks(
                x_train_tensor,
                y_train_tensor,
                x_test_tensor,
                y_test_tensor,
                num_tasks,
                batch_size,
            )
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    @abstractmethod
    def _create_permutation_tasks(
        self,
        x_train_tensor: torch.Tensor,
        y_train_tensor: torch.Tensor,
        x_test_tensor: torch.Tensor,
        y_test_tensor: torch.Tensor,
        num_tasks: int,
        batch_size: int,
    ) -> Tuple[List[DataLoader], List[DataLoader]]:
        """Create permutation-based tasks."""
        pass

    @abstractmethod
    def _create_rotation_tasks(
        self,
        x_train_tensor: torch.Tensor,
        y_train_tensor: torch.Tensor,
        x_test_tensor: torch.Tensor,
        y_test_tensor: torch.Tensor,
        num_tasks: int,
        batch_size: int,
    ) -> Tuple[List[DataLoader], List[DataLoader]]:
        """Create rotation-based tasks."""
        pass

    @abstractmethod
    def _create_class_split_tasks(
        self,
        x_train_tensor: torch.Tensor,
        y_train_tensor: torch.Tensor,
        x_test_tensor: torch.Tensor,
        y_test_tensor: torch.Tensor,
        num_tasks: int,
        batch_size: int,
    ) -> Tuple[List[DataLoader], List[DataLoader]]:
        """Create class-split-based tasks."""
        pass


def load_dataset(dataset_name: str, path_override: str = None) -> DatasetLoader:
    """Factory function to load the appropriate dataset loader.

    Args:
        dataset_name: Name of the dataset ('mnist', 'fashion_mnist', etc.)
        path_override: If given, overrides the path to locate the dataset (for SBATCH)

    Returns:
        DatasetLoader: Appropriate dataset loader instance
    """
    if dataset_name.lower() == "mnist":
        try:
            from mnist import MnistDatasetLoader
        except ImportError:
            from .mnist import MnistDatasetLoader
        return MnistDatasetLoader(path_override)
    elif dataset_name.lower() == "fashion_mnist":
        try:
            from mnist_fashion import FashionMnistDatasetLoader
        except ImportError:
            from .mnist_fashion import FashionMnistDatasetLoader
        return FashionMnistDatasetLoader(path_override)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
