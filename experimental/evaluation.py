# Written by Claude 4
# Edited by Abigail Dodd

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.functional as TF
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from pathlib import Path


class TAGEMEvaluator:
    """
    Evaluator for TA-AGEM models
    Handles test data preparation, evaluation, and reporting
    """

    def __init__(self, params, test_dataloaders=None, save_dir="./test_results"):
        """
        Initialize evaluator with training parameters

        Args:
            params (dict): Training parameters dictionary
            save_dir (str): Directory to save evaluation results
        """
        self.params = params
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.test_dataloaders = test_dataloaders
        self.permutations = params['permutations']

    def prepare_test_data(self, batch_size=100):
        """
        Prepare test data with the same transformations as training tasks

        Args:
            batch_size (int): Batch size for test data loaders

        Returns:
            list: List of test DataLoader objects, one for each task
        """
        # Import here to avoid circular imports
        from mnist import MnistDataloader, training_images_filepath, training_labels_filepath, test_images_filepath, \
            test_labels_filepath

        if self.test_dataloaders is not None:
            # print("Using pre-prepared test dataloaders")
            return self.test_dataloaders

        # Load test data
        _mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath,
                                            test_images_filepath, test_labels_filepath)
        (x_train, y_train), (x_test, y_test) = _mnist_dataloader.load_data()

        '''
        # Convert test data to tensors
        x_test_array = np.array(x_test)
        y_test_array = np.array(y_test)
        x_test_tensor = torch.FloatTensor(x_test_array) / 255.0
        y_test_tensor = torch.LongTensor(y_test_array)
        '''

        # For quick testing, use subset of test data
        if self.params.get('quick_test_mode', False):
            # Use first 1000 test samples for quick testing
            x_test = x_test[:1000]
            y_test = y_test[:1000]

        return self._apply_task_transformations(x_test, y_test, batch_size)

    def _apply_task_transformations(self, x_test_tensor, y_test_tensor, batch_size):
        """Apply task-specific transformations to test data"""
        test_dataloaders = []
        task_type = self.params['task_type']
        num_tasks = self.params['num_tasks']
        num_pixels = self.params['input_dim']

        if task_type == 'permutation':
            # Use the provided permutations if available, otherwise generate new ones (less ideal for evaluation)
            if self.permutations is None or len(self.permutations) != num_tasks:
                print(
                    "Warning: Permutations not provided or count mismatch. Generating new permutations for evaluation.")
                torch.manual_seed(42)  # Ensure reproducibility if generating new ones
                generated_permutations = [torch.randperm(num_pixels) for _ in range(num_tasks)]
                perms_to_use = generated_permutations
            else:
                perms_to_use = self.permutations

            for task_id in range(num_tasks):
                perm = perms_to_use[task_id]
                x_task = x_test_tensor.view(-1, num_pixels)
                x_task = x_task[:, perm]

                dataset = TensorDataset(x_task, y_test_tensor)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                test_dataloaders.append(dataloader)

        elif task_type == 'rotation':
            angles = [i * (360 / num_tasks) for i in range(num_tasks)]

            for task_id in range(num_tasks):
                angle = angles[task_id]
                x_task = x_test_tensor.unsqueeze(1)  # Add channel dimension

                rotated_images = []
                for img in x_task:
                    rotated_img = TF.rotate(img, angle)
                    rotated_images.append(rotated_img)

                x_task = torch.stack(rotated_images)
                x_task = x_task.view(x_task.size(0), -1)  # Flatten

                dataset = TensorDataset(x_task, y_test_tensor)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                test_dataloaders.append(dataloader)

        elif task_type == 'class_split':
            if 10 % num_tasks != 0:
                raise ValueError(f"num_tasks ({num_tasks}) must evenly divide 10 for class_split")

            classes_per_task = 10 // num_tasks

            for task_id in range(num_tasks):
                start_class = task_id * classes_per_task
                end_class = start_class + classes_per_task
                task_classes = list(range(start_class, end_class))

                # Filter test data for this task's classes
                task_mask = torch.zeros(len(y_test_tensor), dtype=torch.bool)
                for cls in task_classes:
                    task_mask |= (y_test_tensor == cls)

                x_task = x_test_tensor[task_mask]
                y_task = y_test_tensor[task_mask]
                x_task = x_task.view(x_task.size(0), -1)  # Flatten

                dataset = TensorDataset(x_task, y_task)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                test_dataloaders.append(dataloader)

        return test_dataloaders

    def evaluate_model(self, model, test_dataloaders, device='cpu'):
        """
        Comprehensive evaluation of the model on test data

        Args:
            model: Trained PyTorch model
            test_dataloaders: List of test data loaders for each task
            device: Device to run evaluation on

        Returns:
            dict: Comprehensive evaluation results
        """
        model.eval()
        model.to(device)

        results = {
            'task_accuracies': [],
            'task_predictions': [],  # Will store numpy arrays for sklearn compatibility
            'task_true_labels': [],  # Will store numpy arrays for sklearn compatibility
            'overall_accuracy': 0.0,
            'per_class_accuracy': {},
            'confusion_matrices': [],
            'evaluation_params': self.params.copy(),
            'timestamp': self.timestamp
        }

        all_predictions_list = []  # Store as list of tensors initially
        all_true_labels_list = []  # Store as list of tensors initially

        print("Evaluating model on test data...")

        with torch.no_grad():
            for task_id, test_dataloader in enumerate(test_dataloaders):
                print(f"Evaluating Task {task_id}...")

                task_predictions_tensors = []  # Collect predictions as tensors
                task_true_labels_tensors = []  # Collect true labels as tensors
                correct = 0
                total = 0

                for batch_data, batch_labels in test_dataloader:
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)

                    # Get model predictions
                    outputs = model(batch_data)
                    _, predicted = torch.max(outputs.data, 1)

                    # Collect predictions and labels as tensors
                    task_predictions_tensors.append(predicted.cpu())
                    task_true_labels_tensors.append(batch_labels.cpu())

                    # Calculate accuracy
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()

                task_accuracy = correct / total if total > 0 else 0.0
                results['task_accuracies'].append(task_accuracy)

                # Convert to numpy arrays ONCE per task for storage and later sklearn use
                task_predictions_np = torch.cat(
                    task_predictions_tensors).numpy() if task_predictions_tensors else np.array([])
                task_true_labels_np = torch.cat(
                    task_true_labels_tensors).numpy() if task_true_labels_tensors else np.array([])

                results['task_predictions'].append(task_predictions_np)
                results['task_true_labels'].append(task_true_labels_np)

                # Add to overall results (still as tensors for efficient concatenation)
                all_predictions_list.extend(task_predictions_tensors)
                all_true_labels_list.extend(task_true_labels_tensors)

                # Generate confusion matrix for this task
                if len(set(task_true_labels_np)) > 1:  # Use numpy array for set conversion
                    cm = confusion_matrix(task_true_labels_np, task_predictions_np)
                    results['confusion_matrices'].append(cm)

                print(f"Task {task_id} Accuracy: {task_accuracy:.4f}")

        # Concatenate all predictions and true labels into single tensors, then convert to numpy ONCE
        if all_true_labels_list:
            all_predictions_tensor = torch.cat(all_predictions_list)
            all_true_labels_tensor = torch.cat(all_true_labels_list)

            # Calculate overall accuracy directly with tensors
            results['overall_accuracy'] = (all_predictions_tensor == all_true_labels_tensor).float().mean().item()

            # Convert to numpy for sklearn functions and per-class accuracy
            all_predictions_np = all_predictions_tensor.numpy()
            all_true_labels_np = all_true_labels_tensor.numpy()

            # Per-class accuracy
            unique_classes = sorted(np.unique(all_true_labels_np))  # Use numpy unique
            for cls in unique_classes:
                cls_mask = (all_true_labels_np == cls)
                if cls_mask.sum() > 0:
                    cls_accuracy = np.sum((all_predictions_np[cls_mask]) == cls) / cls_mask.sum()
                    results['per_class_accuracy'][cls] = cls_accuracy

            results['all_predictions'] = all_predictions_np
            results['all_true_labels'] = all_true_labels_np
        else:
            results['all_predictions'] = np.array([])
            results['all_true_labels'] = np.array([])

        return results

    def print_evaluation_report(self, results):
        """Print a comprehensive evaluation report"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE TEST EVALUATION REPORT")
        print("=" * 60)

        print(f"Task Type: {self.params['task_type']}")
        print(f"Overall Test Accuracy: {results['overall_accuracy']:.4f}")
        print(f"Number of Tasks: {len(results['task_accuracies'])}")

        print("\n--- Per-Task Accuracies ---")
        for i, acc in enumerate(results['task_accuracies']):
            print(f"Task {i}: {acc:.4f}")

        print("\n--- Per-Class Accuracies ---")
        for cls, acc in results['per_class_accuracy'].items():
            print(f"Class {cls}: {acc:.4f}")

        # Calculate catastrophic forgetting metrics
        if len(results['task_accuracies']) > 1:
            first_task_acc = results['task_accuracies'][0]
            last_task_acc = results['task_accuracies'][-1]
            forgetting = first_task_acc - last_task_acc
            print(f"\n--- Catastrophic Forgetting Analysis ---")
            print(f"First Task Final Accuracy: {first_task_acc:.4f}")
            print(f"First Task Accuracy After All Tasks: {last_task_acc:.4f}")
            print(f"Forgetting (First - Last): {forgetting:.4f}")

        # Classification report for overall performance
        if len(results['all_true_labels']) > 0:
            print("\n--- Detailed Classification Report ---")
            print(classification_report(results['all_true_labels'], results['all_predictions']))

    def plot_evaluation_results(self, results, save_plots=True):
        """Create visualizations of the evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'TA-AGEM Test Evaluation - {self.params["task_type"].title()} Tasks', fontsize=16)

        # 1. Task accuracies over time
        axes[0, 0].plot(range(len(results['task_accuracies'])), results['task_accuracies'], 'bo-')
        axes[0, 0].set_title('Accuracy per Task')
        axes[0, 0].set_xlabel('Task ID')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True)

        # 2. Per-class accuracies
        if results['per_class_accuracy']:
            classes = list(results['per_class_accuracy'].keys())
            accuracies = list(results['per_class_accuracy'].values())
            axes[0, 1].bar(classes, accuracies)
            axes[0, 1].set_title('Per-Class Accuracy')
            axes[0, 1].set_xlabel('Class')
            axes[0, 1].set_ylabel('Accuracy')

        # 3. Overall confusion matrix
        if len(results['all_true_labels']) > 0:
            cm = confusion_matrix(results['all_true_labels'], results['all_predictions'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues')
            axes[1, 0].set_title('Overall Confusion Matrix')
            axes[1, 0].set_xlabel('Predicted')
            axes[1, 0].set_ylabel('True')

        # 4. Accuracy distribution
        axes[1, 1].hist(results['task_accuracies'], bins=min(10, len(results['task_accuracies'])),
                        alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Distribution of Task Accuracies')
        axes[1, 1].set_xlabel('Accuracy')
        axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()

        if save_plots:
            plot_path = self.save_dir / f"evaluation_plots_{self.timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {plot_path}")

        plt.show()

    def save_results(self, results):
        """Save evaluation results to pickle file"""
        results_path = self.save_dir / f"test_results_{self.timestamp}.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Test results saved to {results_path}")
        return results_path

    def run_full_evaluation(self, model, device='cpu', save_results=True):
        """
        Run complete evaluation pipeline

        Args:
            model: Trained PyTorch model
            device: Device to run evaluation on
            save_results: Whether to save results and plots

        Returns:
            dict: Comprehensive evaluation results
        """
        print("\n" + "=" * 50)
        print("STARTING COMPREHENSIVE TEST EVALUATION")
        print("=" * 50)

        # Prepare test data
        test_dataloaders = self.prepare_test_data()

        # Evaluate model
        results = self.evaluate_model(model, test_dataloaders, device)

        # Print report
        self.print_evaluation_report(results)

        # Create plots
        self.plot_evaluation_results(results, save_plots=save_results)

        # Save results
        if save_results:
            self.save_results(results)

        return results


# Utility functions for standalone use
def quick_evaluate_model(model, params, device='cpu'):
    """
    Quick evaluation function for immediate use

    Args:
        model: Trained PyTorch model
        params: Training parameters dictionary
        device: Device to run evaluation on

    Returns:
        dict: Evaluation results
    """
    evaluator = TAGEMEvaluator(params)
    return evaluator.run_full_evaluation(model, device)


def compare_models(models, model_names, params, device='cpu'):
    """
    Compare multiple models on the same test data

    Args:
        models: List of trained PyTorch models
        model_names: List of names for the models
        params: Training parameters dictionary
        device: Device to run evaluation on

    Returns:
        dict: Comparison results
    """
    evaluator = TAGEMEvaluator(params)
    test_dataloaders = evaluator.prepare_test_data()

    comparison_results = {}

    for model, name in zip(models, model_names):
        print(f"\nEvaluating {name}...")
        results = evaluator.evaluate_model(model, test_dataloaders, device)
        comparison_results[name] = results

    # Print comparison summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)

    for name, results in comparison_results.items():
        print(f"{name}: Overall Accuracy = {results['overall_accuracy']:.4f}")

    return comparison_results
