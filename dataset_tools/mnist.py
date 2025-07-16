# This loads and visualizes MNIST data.
# Original here: https://www.kaggle.com/code/hojjatk/read-mnist-dataset
# Modified!
import numpy as np  # linear algebra
import struct
from array import array
from os.path import join
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.functional as TF

try:
    from .load_dataset import DatasetLoader
    from .dataset_utils import get_dataset_path
except ImportError:
    from load_dataset import DatasetLoader
    from dataset_utils import get_dataset_path


#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(
        self,
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    "Magic number mismatch, expected 2049, got {}".format(magic)
                )
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch, expected 2051, got {}".format(magic)
                )
            image_data = array("B", file.read())

        # Convert to tensor format directly
        images_array = np.array(image_data, dtype=np.uint8).reshape(size, rows, cols)
        labels_array = np.array(labels, dtype=np.uint8)

        # Convert to PyTorch tensors
        images_tensor = torch.FloatTensor(images_array)
        labels_tensor = torch.LongTensor(labels_array)

        return images_tensor, labels_tensor

    def load_data(self):
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath
        )
        return (x_train, y_train), (x_test, y_test)


class MnistDatasetLoader(DatasetLoader):
    """MNIST dataset loader implementing the DatasetLoader interface."""

    def __init__(self, path_override=None):
        super().__init__()
        # Set file paths based on MNIST datasets
        path = get_dataset_path("MNIST", "hojjatk/mnist-dataset", path_override=path_override)
        self.training_images_filepath = join(
            path, "train-images-idx3-ubyte/train-images-idx3-ubyte"
        )
        self.training_labels_filepath = join(
            path, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"
        )
        self.test_images_filepath = join(
            path, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
        )
        self.test_labels_filepath = join(
            path, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"
        )

    def load_raw_data(self):
        """Load raw MNIST data."""
        mnist_loader = MnistDataloader(
            self.training_images_filepath,
            self.training_labels_filepath,
            self.test_images_filepath,
            self.test_labels_filepath,
        )
        (x_train, y_train), (x_test, y_test) = mnist_loader.load_data()
        return x_train, y_train, x_test, y_test

    def preprocess_data(self, x_train, y_train, x_test, y_test, quick_test=False):
        """Preprocess MNIST data into PyTorch tensors."""
        # Data is already in tensor format, just normalize
        x_train_tensor = x_train / 255.0
        y_train_tensor = y_train
        x_test_tensor = x_test / 255.0
        y_test_tensor = y_test

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
    ):
        """Create permutation-based tasks for MNIST."""
        train_dataloaders = []
        test_dataloaders = []
        num_pixels = 28 * 28

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
                train_dataset, batch_size=batch_size, shuffle=True
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
    ):
        """Create rotation-based tasks for MNIST."""
        train_dataloaders = []
        test_dataloaders = []
        angles = [i * (360 / num_tasks) for i in range(num_tasks)]

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
            x_train_task = x_train_subset.unsqueeze(1)
            rotated_train_images = []
            for img in x_train_task:
                rotated_img = TF.rotate(img, angle)
                rotated_train_images.append(rotated_img)
            x_train_task = torch.stack(rotated_train_images)
            x_train_task = x_train_task.view(x_train_task.size(0), -1)

            # Rotate test images
            x_test_task = x_test_subset.unsqueeze(1)
            rotated_test_images = []
            for img in x_test_task:
                rotated_img = TF.rotate(img, angle)
                rotated_test_images.append(rotated_img)
            x_test_task = torch.stack(rotated_test_images)
            x_test_task = x_test_task.view(x_test_task.size(0), -1)

            # Create datasets and dataloaders
            train_dataset = TensorDataset(x_train_task, y_train_subset)
            test_dataset = TensorDataset(x_test_task, y_test_subset)
            train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            test_dataloader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )
            train_dataloaders.append(train_dataloader)
            test_dataloaders.append(test_dataloader)

        return train_dataloaders, test_dataloaders

    def _create_class_split_tasks(
        self,
        x_train_tensor,
        y_train_tensor,
        x_test_tensor,
        y_test_tensor,
        num_tasks,
        batch_size,
    ):
        """Create class-split-based tasks for MNIST."""
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
            x_train_task = x_train_task.view(x_train_task.size(0), -1)  # (N, 784)
            x_test_task = x_test_task.view(x_test_task.size(0), -1)  # (N, 784)

            # Create datasets and dataloaders
            train_dataset = TensorDataset(x_train_task, y_train_task)
            test_dataset = TensorDataset(x_test_task, y_test_task)
            train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            test_dataloader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )
            train_dataloaders.append(train_dataloader)
            test_dataloaders.append(test_dataloader)

        return train_dataloaders, test_dataloaders


# Backward compatibility function
# def prepare_domain_incremental_mnist(task_type, num_tasks, batch_size, quick_test=False):
#     """Load and prepare MNIST data for domain-incremental learning (backward compatibility)."""
#     loader = MnistDatasetLoader()
#     return loader.prepare_domain_incremental_data(task_type, num_tasks, batch_size, quick_test)


if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt

    #
    # Load MNIST dataset
    #
    # Set file paths based on added MNIST Datasets
    input_path = get_dataset_path("MNIST", "hojjatk/mnist-dataset")
    print(f"Dataset is at {input_path}")
    training_images_filepath = join(
        input_path, "train-images-idx3-ubyte/train-images-idx3-ubyte"
    )
    training_labels_filepath = join(
        input_path, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"
    )
    test_images_filepath = join(
        input_path, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
    )
    test_labels_filepath = join(
        input_path, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"
    )

    _mnist_dataloader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    )
    (x_train, y_train), (x_test, y_test) = _mnist_dataloader.load_data()

    #
    # Helper function to show a list of images with their relating titles
    #
    def show_images(images, title_texts):
        cols = 5
        rows = int(len(images) / cols) + 1
        plt.figure(figsize=(30, 20))
        index = 1
        for x in zip(images, title_texts):
            image = x[0]
            title_text = x[1]
            plt.subplot(rows, cols, index)
            plt.imshow(image, cmap="gray")
            if title_text != "":
                plt.title(title_text, fontsize=15)
            index += 1

    #
    # Show some random training and test images
    #
    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = random.randint(1, len(x_train) - 1)
        images_2_show.append(x_train[r].numpy())
        titles_2_show.append(
            "training image [" + str(r) + "] = " + str(y_train[r].item())
        )

    for i in range(0, 5):
        r = random.randint(1, len(x_test) - 1)
        images_2_show.append(x_test[r].numpy())
        titles_2_show.append("test image [" + str(r) + "] = " + str(y_test[r].item()))

    show_images(images_2_show, titles_2_show)
