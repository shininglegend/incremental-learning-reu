"""
CIFAR-10 dataset loader implementing the DatasetLoader interface.
This module provides functionality to load CIFAR-10 data and prepare it
for incremental learning tasks with grayscale conversion.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.functional as TF

try:
    from .load_dataset import DatasetLoader
    from .dataset_utils import get_dataset_path
except ImportError:
    from load_dataset import DatasetLoader
    from dataset_utils import get_dataset_path


class Cifar10Dataloader(object):
    """Low-level CIFAR-10 data loader."""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_batch(self, batch_file):
        """Load a single CIFAR-10 batch file."""
        with open(batch_file, "rb") as file:
            batch = pickle.load(file, encoding="latin1")

        # Reshape from flat array to (N, 3, 32, 32) then transpose to (N, 32, 32, 3)
        X = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        y = np.array(batch["labels"])
        return X, y

    def load_data(self):
        """Load all CIFAR-10 training and test data."""
        # Load training batches
        x_train_list = []
        y_train_list = []
        for i in range(1, 6):
            batch_file = os.path.join(self.data_dir, f"data_batch_{i}")
            X_batch, y_batch = self.load_batch(batch_file)
            x_train_list.append(X_batch)
            y_train_list.append(y_batch)

        x_train = np.concatenate(x_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        # Load test batch
        test_batch_file = os.path.join(self.data_dir, "test_batch")
        x_test, y_test = self.load_batch(test_batch_file)

        return x_train, y_train, x_test, y_test


class Cifar10DatasetLoader(DatasetLoader):
    """CIFAR-10 dataset loader implementing the DatasetLoader interface."""

    def __init__(self, path_override=None):
        super().__init__()
        path = get_dataset_path(
            "CIFAR10", "pankrzysiu/cifar10-python", path_override=path_override
        )
        self.data_dir = os.path.join(path, "cifar-10-batches-py")

    def load_raw_data(self):
        """Load raw CIFAR-10 data."""
        cifar_loader = Cifar10Dataloader(self.data_dir)
        x_train, y_train, x_test, y_test = cifar_loader.load_data()
        return x_train, y_train, x_test, y_test

    def _convert_to_grayscale(self, images):
        """Convert RGB images to grayscale using PIL."""
        grey_images = []
        for image in images:
            img = Image.fromarray(image.astype(np.uint8))
            img_grey = img.convert("L")
            grey_images.append(np.array(img_grey))
        return np.array(grey_images)

    def preprocess_data(self, x_train, y_train, x_test, y_test, quick_test=False):
        """Preprocess CIFAR-10 data into PyTorch tensors with grayscale conversion."""
        # Convert to grayscale
        x_train_grey = self._convert_to_grayscale(x_train)
        x_test_grey = self._convert_to_grayscale(x_test)

        # Convert to tensors and normalize
        x_train_tensor = torch.FloatTensor(x_train_grey) / 255.0
        y_train_tensor = torch.LongTensor(y_train)
        x_test_tensor = torch.FloatTensor(x_test_grey) / 255.0
        y_test_tensor = torch.LongTensor(y_test)

        # For quick testing, use only first 1000 samples per class
        if quick_test:
            quick_train_indices = []
            quick_test_indices = []
            for class_id in range(10):
                # Training data
                class_mask = y_train_tensor == class_id
                class_indices = torch.where(class_mask)[0][:1000]
                quick_train_indices.extend(class_indices.tolist())

                # Test data
                class_mask = y_test_tensor == class_id
                class_indices = torch.where(class_mask)[0][:200]
                quick_test_indices.extend(class_indices.tolist())

            quick_train_indices = torch.tensor(quick_train_indices)
            quick_test_indices = torch.tensor(quick_test_indices)
            x_train_tensor = x_train_tensor[quick_train_indices]
            y_train_tensor = y_train_tensor[quick_train_indices]
            x_test_tensor = x_test_tensor[quick_test_indices]
            y_test_tensor = y_test_tensor[quick_test_indices]

        return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor

    def _create_permutation_tasks(
        self,
        x_train_tensor,
        y_train_tensor,
        x_test_tensor,
        y_test_tensor,
        num_tasks,
        batch_size,
        use_cuda,
    ):
        """Create permutation-based tasks for CIFAR-10."""
        train_dataloaders = []
        test_dataloaders = []
        num_pixels = 32 * 32

        # Split data into disjoint subsets for each task
        train_size_per_task = len(x_train_tensor) // num_tasks
        test_size_per_task = len(x_test_tensor) // num_tasks

        for task_id in range(num_tasks):
            # Get disjoint data subset for this task
            train_start = task_id * train_size_per_task
            train_end = (
                (task_id + 1) * train_size_per_task
                if task_id < num_tasks - 1
                else len(x_train_tensor)
            )
            test_start = task_id * test_size_per_task
            test_end = (
                (task_id + 1) * test_size_per_task
                if task_id < num_tasks - 1
                else len(x_test_tensor)
            )

            x_train_subset = x_train_tensor[train_start:train_end]
            y_train_subset = y_train_tensor[train_start:train_end]
            x_test_subset = x_test_tensor[test_start:test_end]
            y_test_subset = y_test_tensor[test_start:test_end]

            # Generate a random permutation for this task
            perm = torch.randperm(num_pixels)

            # Flatten images and apply permutation - TRAIN
            x_train_task = x_train_subset.view(-1, num_pixels)
            x_train_task = x_train_task[:, perm]

            # Flatten images and apply permutation - TEST
            x_test_task = x_test_subset.view(-1, num_pixels)
            x_test_task = x_test_task[:, perm]

            # Create datasets and dataloaders
            train_dataset = TensorDataset(x_train_task, y_train_subset)
            test_dataset = TensorDataset(x_test_task, y_test_subset)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=use_cuda,
            )
            test_dataloader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )
            train_dataloaders.append(train_dataloader)
            test_dataloaders.append(test_dataloader)

        return train_dataloaders, test_dataloaders

    def _create_rotation_tasks(
        self,
        x_train_tensor,
        y_train_tensor,
        x_test_tensor,
        y_test_tensor,
        num_tasks,
        batch_size,
        use_cuda,
    ):
        """Create rotation-based tasks for CIFAR-10."""
        train_dataloaders = []
        test_dataloaders = []

        angles = [i * 20 for i in range(num_tasks)]

        # Split data into disjoint subsets for each task
        train_size_per_task = len(x_train_tensor) // num_tasks
        test_size_per_task = len(x_test_tensor) // num_tasks

        for task_id in range(num_tasks):
            # Get disjoint data subset for this task
            train_start = task_id * train_size_per_task
            train_end = (
                (task_id + 1) * train_size_per_task
                if task_id < num_tasks - 1
                else len(x_train_tensor)
            )
            test_start = task_id * test_size_per_task
            test_end = (
                (task_id + 1) * test_size_per_task
                if task_id < num_tasks - 1
                else len(x_test_tensor)
            )

            x_train_subset = x_train_tensor[train_start:train_end]
            y_train_subset = y_train_tensor[train_start:train_end]
            x_test_subset = x_test_tensor[test_start:test_end]
            y_test_subset = y_test_tensor[test_start:test_end]

            angle = angles[task_id]

            # Rotate train images
            rotated_train_images = []
            for img in x_train_subset:
                rotated_img = TF.rotate(img.unsqueeze(0), angle, fill=[0.0]).squeeze(0)
                rotated_train_images.append(rotated_img)
            x_train_task = torch.stack(rotated_train_images)
            x_train_task = x_train_task.view(x_train_task.size(0), -1)

            # Rotate test images
            rotated_test_images = []
            for img in x_test_subset:
                rotated_img = TF.rotate(img.unsqueeze(0), angle, fill=[0.0]).squeeze(0)
                rotated_test_images.append(rotated_img)
            x_test_task = torch.stack(rotated_test_images)
            x_test_task = x_test_task.view(x_test_task.size(0), -1)

            # Create datasets and dataloaders
            train_dataset = TensorDataset(x_train_task, y_train_subset)
            test_dataset = TensorDataset(x_test_task, y_test_subset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=use_cuda,
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )

            train_dataloaders.append(train_loader)
            test_dataloaders.append(test_loader)

        return train_dataloaders, test_dataloaders

    def _create_class_split_tasks(
        self,
        x_train_tensor,
        y_train_tensor,
        x_test_tensor,
        y_test_tensor,
        num_tasks,
        batch_size,
        use_cuda,
    ):
        """Create class-split-based tasks for CIFAR-10."""
        train_dataloaders = []
        test_dataloaders = []

        # Check if num_tasks is 5 (for 2 classes per task)
        if num_tasks != 5:
            raise ValueError(
                f"num_tasks ({num_tasks}) must be 5 for class_split (2 classes per task)"
            )

        for task_id in range(num_tasks):
            # Determine which classes belong to this task (2 classes per task)
            start_class = task_id * 2
            end_class = start_class + 2
            task_classes = list(range(start_class, end_class))

            # Filter train data for this task's classes
            train_task_mask = torch.zeros(len(y_train_tensor), dtype=torch.bool)
            for cls in task_classes:
                train_task_mask |= y_train_tensor == cls

            x_train_task = x_train_tensor[train_task_mask]
            y_train_task = y_train_tensor[train_task_mask]

            # Filter test data for this task's classes
            test_task_mask = torch.zeros(len(y_test_tensor), dtype=torch.bool)
            for cls in task_classes:
                test_task_mask |= y_test_tensor == cls

            x_test_task = x_test_tensor[test_task_mask]
            y_test_task = y_test_tensor[test_task_mask]

            # Remap labels: even labels -> 0, odd labels -> 1
            y_train_task = y_train_task % 2
            y_test_task = y_test_task % 2

            # Flatten for consistent model input
            x_train_task = x_train_task.view(x_train_task.size(0), -1)
            x_test_task = x_test_task.view(x_test_task.size(0), -1)

            # Create datasets and dataloaders
            train_dataset = TensorDataset(x_train_task, y_train_task)
            test_dataset = TensorDataset(x_test_task, y_test_task)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=use_cuda,
            )
            test_dataloader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )
            train_dataloaders.append(train_dataloader)
            test_dataloaders.append(test_dataloader)

        return train_dataloaders, test_dataloaders


def show_images(images, title_texts, figure_title="", img_size=32):
    """Display CIFAR-10 images in a grid."""
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(15, 10))
    if figure_title:
        plt.suptitle(figure_title, fontsize=16)
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(
            image.reshape(img_size, img_size) if len(image.shape) == 1 else image,
            cmap="gray",
        )
        if title_text != "":
            plt.title(title_text, fontsize=10)
        plt.axis("off")
        index += 1
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """Show sample original and grayscale CIFAR-10 images."""
    # Load a small sample for demonstration
    loader = Cifar10DatasetLoader()
    x_train, y_train, x_test, y_test = loader.load_raw_data()

    # Print the actual shape
    print(f"CIFAR-10 image shape: {x_train[0].shape}")  # Should be (32, 32, 3)
    print(f"Total pixels per image: {x_train[0].size}")  # Should be 3,072

    # Show original color images
    plt.figure(figsize=(10, 2))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(x_train[i])
        plt.axis("off")
    plt.suptitle("Original CIFAR-10 Images")
    plt.show()

    # Convert to grayscale and show
    x_train_grey = loader._convert_to_grayscale(x_train[:5])
    plt.figure(figsize=(10, 2))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(x_train_grey[i], cmap="gray")
        plt.axis("off")
    plt.suptitle("Grayscale CIFAR-10 Images")
    plt.show()

    # Show the task images.
    loader = Cifar10DatasetLoader()

    # Load and preprocess data
    x_train, y_train, x_test, y_test = loader.load_raw_data()
    x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = (
        loader.preprocess_data(x_train, y_train, x_test, y_test, quick_test=True)
    )

    # Show permutation task samples
    train_loaders, _ = loader._create_permutation_tasks(
        x_train_tensor,
        y_train_tensor,
        x_test_tensor,
        y_test_tensor,
        num_tasks=3,
        batch_size=64,
        use_cuda=False,
    )

    sample_images = []
    titles = []
    for i, train_loader in enumerate(train_loaders):
        batch_x, batch_y = next(iter(train_loader))
        sample_images.append(batch_x[0].numpy())
        titles.append(f"Permutation Task {i+1}")

    show_images(sample_images, titles, "Permutation Tasks Sample", img_size=32)

    # Show rotation task samples
    train_loaders, _ = loader._create_rotation_tasks(
        x_train_tensor,
        y_train_tensor,
        x_test_tensor,
        y_test_tensor,
        num_tasks=10,
        batch_size=64,
        use_cuda=False,
    )

    sample_images = []
    titles = []
    for i, train_loader in enumerate(train_loaders):
        if i > 3:
            break
        batch_x, batch_y = next(iter(train_loader))
        sample_images.append(batch_x[0].numpy())
        titles.append(f"Rotation Task {i+1}")

    show_images(sample_images, titles, "Rotation Tasks Sample", img_size=32)
    print("Done.")
