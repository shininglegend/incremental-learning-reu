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
from config import params


class TAGEMEvaluator:
    """
    Comprehensive evaluator for TA-AGEM models
    Handles test data preparation, evaluation, and reporting
    """

    def __init__(self, test_dataloaders=None, save_dir="./test_results"):
        """
        Initialize evaluator with training parameters

        Args:
            params (dict): Training parameters dictionary
            test_dataloaders (list): Pre-prepared list of test DataLoaders from dataset loader
            save_dir (str): Directory to save evaluation results
        """
        self.params = params
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.test_dataloaders = test_dataloaders

    def prepare_test_data(self):
        """
        Return pre-prepared test dataloaders.
        """
        if self.test_dataloaders is not None:
            print("Using pre-prepared test dataloaders from dataset loader")
            return self.test_dataloaders
        else:
            raise ValueError(
                "No test_dataloaders provided to evaluator. "
                "Please pass test_dataloaders from your dataset loader to ensure "
                "consistent transformations between training and testing data."
            )

    # Remove _apply_task_transformations method entirely

    def run_full_evaluation(self, model, device='cpu', save_results=True, intermediate_eval_history=None):
        """
        Run complete evaluation pipeline

        Args:
            model: Trained PyTorch model
            device: Device to run evaluation on
            save_results: Whether to save results and plots
            intermediate_eval_history (list, optional): History of accuracies from intermediate evaluations.

        Returns:
            dict: Comprehensive evaluation results
        """
        print("\n" + "=" * 50)
        print("STARTING COMPREHENSIVE TEST EVALUATION")
        print("=" * 50)

        # Use pre-prepared test dataloaders (required)
        test_dataloaders = self.prepare_test_data()

        # Evaluate model on all tasks (final evaluation)
        results = self.evaluate_model(model, test_dataloaders, device)

        # Print report
        self.print_evaluation_report(results, intermediate_eval_history)

        # Create plots, passing the intermediate history
        self.plot_evaluation_results(results, intermediate_eval_history, save_plots=save_results)

        # Save results
        if save_results:
            self.save_results(results)

        return results

    def evaluate_model(self, model, test_dataloaders, device='cpu'):
        """
        Comprehensive evaluation of the model on test data for all provided dataloaders.
        This is typically used for the final evaluation after all tasks are trained.

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
            'timestamp': self.timestamp,
            'all_predictions': np.array([]), # Initialize for safety
            'all_true_labels': np.array([])  # Initialize for safety
        }

        all_predictions_list = []  # Store as list of tensors initially
        all_true_labels_list = []  # Store as list of tensors initially

        print("Evaluating model on test data...")

        with torch.no_grad():
            for task_id, test_dataloader in enumerate(test_dataloaders):
                print(f"  Evaluating Task {task_id}...")

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

                # Generate confusion matrix for this task (if enough samples/classes)
                if len(set(task_true_labels_np)) > 1 and len(task_true_labels_np) > 0:
                    cm = confusion_matrix(task_true_labels_np, task_predictions_np)
                    results['confusion_matrices'].append(cm)

                print(f"  Task {task_id} Accuracy: {task_accuracy:.4f}")

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
            unique_classes = sorted(np.unique(all_true_labels_np))
            for cls in unique_classes:
                cls_mask = (all_true_labels_np == cls)
                if cls_mask.sum() > 0:
                    cls_accuracy = np.sum((all_predictions_np[cls_mask]) == cls) / cls_mask.sum()
                    results['per_class_accuracy'][cls] = cls_accuracy

            results['all_predictions'] = all_predictions_np
            results['all_true_labels'] = all_true_labels_np
        else:
            results['overall_accuracy'] = 0.0 # No data to evaluate
            results['per_class_accuracy'] = {}

        return results

    def evaluate_tasks_up_to(self, model, current_task_idx, all_test_dataloaders, device='cpu'):
        """
        Evaluates the model on tasks from 0 up to current_task_idx (inclusive).
        This is used for intermediate evaluations during training to track forgetting.

        Args:
            model: Current trained PyTorch model.
            current_task_idx (int): The index of the last task to evaluate (inclusive).
            all_test_dataloaders (list): The full list of all test data loaders.
            device (str): Device to run evaluation on.

        Returns:
            list: A list of accuracies for tasks 0 to current_task_idx.
        """
        model.eval()
        model.to(device)
        accuracies = []

        with torch.no_grad():
            # Evaluate on all tasks seen so far (from 0 up to current_task_idx)
            for task_id in range(current_task_idx + 1):
                test_dataloader = all_test_dataloaders[task_id]
                correct = 0
                total = 0
                for batch_data, batch_labels in test_dataloader:
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)

                    outputs = model(batch_data)
                    _, predicted = torch.max(outputs.data, 1)

                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()

                task_accuracy = correct / total if total > 0 else 0.0
                accuracies.append(task_accuracy)
        return accuracies

    def print_evaluation_report(self, results, intermediate_eval_report):
        """Print a comprehensive evaluation report"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE TEST EVALUATION REPORT (Final Model State)")
        print("=" * 60)

        print(f"Task Type: {self.params['task_type']}")
        print(f"Overall Test Accuracy: {results['overall_accuracy']:.4f}")
        print(f"Number of Tasks Evaluated: {len(results['task_accuracies'])}")

        print("\n--- Final Per-Task Accuracies (after all tasks trained) ---")
        for i, acc in enumerate(results['task_accuracies']):
            print(f"Task {i}: {acc:.4f}")

        print("\n--- Final Per-Class Accuracies (after all tasks trained) ---")
        # Sort by class ID for consistent output
        sorted_per_class_acc = sorted(results['per_class_accuracy'].items())
        for cls, acc in sorted_per_class_acc:
            print(f"Class {cls}: {acc:.4f}")

        # Catastrophic forgetting metrics
        if len(results['task_accuracies']) > 1:
            first_task_first_acc = intermediate_eval_report[0][0]
            first_task_final_acc = results['task_accuracies'][0] # This is the accuracy of the last task learned
            forgetting = first_task_first_acc - first_task_final_acc

            print(f"\n--- Catastrophic Forgetting Analysis (Task 0) ---")
            print(f"Accuracy on Task 0 (after only Task 0 trained): {first_task_first_acc:.4f}")
            print(f"Accuracy on Task 0 (after all tasks trained): {first_task_final_acc:.4f}")
            print(f"Forgetting (Task 0 Final Acc - Last Task Final Acc): {forgetting:.4f}")

        # Classification report for overall performance
        if len(results['all_true_labels']) > 0:
            print("\n--- Detailed Classification Report (Overall) ---")
            # Ensure target_names are provided for better report readability
            target_names = [str(i) for i in sorted(np.unique(results['all_true_labels']))]
            print(classification_report(results['all_true_labels'], results['all_predictions'], target_names=target_names, zero_division=0))

    def plot_evaluation_results(self, results, intermediate_eval_history, save_plots=True):
        """
        Create visualizations of the evaluation results, including forgetting over time.

        Args:
            results (dict): Final evaluation results.
            intermediate_eval_history (list): List of lists, where each inner list contains
                                              accuracies of tasks 0 to N after task N was trained.
            save_plots (bool): Whether to save plots to a file.
        """
        # Adjust subplot layout: 2 rows, 2 columns for now, will adjust based on what plots are kept

        fig, axes = plt.subplots(2, 2, figsize=(18, 14)) # Increased figure size for better visibility
        if params['quick_test_mode']:
            fig.suptitle(f'TA-AGEM Quick Test - {self.params["dataset_name"]}, {self.params["task_type"].title()} Tasks', fontsize=18)
        else:
            fig.suptitle(f'TA-AGEM Test Evaluation - {self.params["dataset_name"]}, {self.params["task_type"].title()} Tasks', fontsize=18)

        # Plot 1: Table of Intermediate Task Accuracies
        axes[0, 0].set_title('Intermediate Task Accuracies Over Training Progress', fontsize=14)
        axes[0, 0].axis('off')  # Turn off axes for this subplot as we are drawing a table

        if intermediate_eval_history:
            num_tasks = self.params['num_tasks']
            num_training_steps = len(intermediate_eval_history)

            # Prepare table data
            # Header row: ["Task ID", "After Task 0", "After Task 1", ..., "After Task N-1", "Average"]
            column_headers = ["Task ID"] + [f"After Task {i}" for i in range(num_training_steps)] + ["Average"]

            table_data = []
            for task_id in range(num_tasks):
                row = [f"Task {task_id}"]
                task_accuracies_at_each_step = []
                for step_idx in range(num_training_steps):
                    if task_id < len(intermediate_eval_history[step_idx]):
                        acc = intermediate_eval_history[step_idx][task_id]
                        row.append(f"{acc:.4f}")
                        task_accuracies_at_each_step.append(acc)
                    else:
                        row.append("N/A")  # Task not yet introduced or evaluated at this step
                # Calculate average for this task across steps it was evaluated
                avg_task_acc = np.mean(task_accuracies_at_each_step) if task_accuracies_at_each_step else np.nan
                row.append(f"{avg_task_acc:.4f}" if not np.isnan(avg_task_acc) else "N/A")
                table_data.append(row)

            # Add a row for average accuracy at each step
            average_row = ["Average"]
            overall_accuracies_at_each_step = []
            for step_idx in range(num_training_steps):
                # Calculate average of all tasks evaluated *at that specific step*
                accuracies_at_step = [acc for task_accs in intermediate_eval_history[step_idx] for acc in [task_accs]]
                if accuracies_at_step:
                    avg_at_step = np.mean(accuracies_at_step)
                    average_row.append(f"{avg_at_step:.4f}")
                    overall_accuracies_at_each_step.append(avg_at_step)
                else:
                    average_row.append("N/A")
            # Final overall average of the model's accuracy across all tasks (this is the results['overall_accuracy'])
            average_row.append(f"{results['overall_accuracy']:.4f}")
            table_data.append(average_row)

            # Create the table
            table = axes[0, 0].table(cellText=table_data,
                                     colLabels=column_headers,
                                     loc='center',
                                     cellLoc='center',
                                     colColours=['#f2f2f2'] * len(column_headers))
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)  # Adjust size if needed for readability
            # Adjust column widths to fit content
            for (row, col), cell in table.get_celld().items():
                cell.set_height(0.08)  # Adjust row height
                if row == 0:  # Header row
                    cell.set_text_props(weight='bold', color='black')
                    cell.set_facecolor('#d9d9d9')
                elif row == len(table_data):  # Average row
                    cell.set_text_props(weight='bold', color='darkblue')
                    cell.set_facecolor('#e6f2ff')
                if col == 0:  # Task ID column
                    cell.set_text_props(weight='bold')

        else:
            axes[0, 0].text(0.5, 0.5, 'Intermediate evaluation history not available for table.',
                            horizontalalignment='center', verticalalignment='center', transform=axes[0, 0].transAxes,
                            fontsize=12, color='gray')

        # Plot 2: Accuracy of each task as new tasks are added (Forgetting Plot)
        if intermediate_eval_history:
            num_tasks = len(intermediate_eval_history[-1]) # Total number of tasks
            x_axis_labels = [f"After Task {i}" for i in range(len(intermediate_eval_history))]

            # Prepare data for plotting: each row is a task, each column is a training step
            # Fill with NaN for tasks not yet seen
            max_len = max(len(h) for h in intermediate_eval_history)
            padded_history = np.full((num_tasks, len(intermediate_eval_history)), np.nan)

            for step_idx, step_accuracies in enumerate(intermediate_eval_history):
                for task_idx, acc in enumerate(step_accuracies):
                    padded_history[task_idx, step_idx] = acc

            for task_idx in range(num_tasks):
                axes[0, 1].plot(x_axis_labels, padded_history[task_idx, :], marker='o',
                                label=f'Task {task_idx}')

            axes[0, 1].set_title('Accuracy of Each Task Over Training Progress', fontsize=14)
            axes[0, 1].set_xlabel('Training Step (After Task X Trained)', fontsize=12)
            axes[0, 1].set_ylabel('Accuracy', fontsize=12)
            axes[0, 1].set_xticks(range(len(x_axis_labels)))
            axes[0, 1].set_xticklabels(x_axis_labels, rotation=45, ha='right')
            axes[0, 1].legend(title="Task ID")
            axes[0, 1].grid(True, linestyle='--', alpha=0.7)
            axes[0, 1].set_ylim([0, 1]) # Ensure y-axis is from 0 to 1
        else:
            axes[0, 1].set_title('Accuracy of Each Task Over Training Progress (No Data)', fontsize=14)
            axes[0, 1].text(0.5, 0.5, 'Intermediate evaluation history not available.',
                            horizontalalignment='center', verticalalignment='center', transform=axes[0, 1].transAxes)


        # Plot 3: Per-class accuracies
        if results['per_class_accuracy']:
            classes = list(results['per_class_accuracy'].keys())
            accuracies = list(results['per_class_accuracy'].values())
            # Sort by class ID for consistent plotting
            sorted_classes_accuracies = sorted(zip(classes, accuracies))
            sorted_classes = [c for c, a in sorted_classes_accuracies]
            sorted_accuracies = [a for c, a in sorted_classes_accuracies]

            axes[1, 0].bar(sorted_classes, sorted_accuracies, color='skyblue')
            axes[1, 0].set_title('Final Per-Class Accuracy', fontsize=14)
            axes[1, 0].set_xlabel('Class', fontsize=12)
            axes[1, 0].set_ylabel('Accuracy', fontsize=12)
            axes[1, 0].set_xticks(sorted_classes)
            axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
            axes[1, 0].set_ylim([0, 1]) # Ensure y-axis is from 0 to 1
        else:
            axes[1, 0].set_title('Final Per-Class Accuracy (No Data)', fontsize=14)
            axes[1, 0].text(0.5, 0.5, 'Per-class accuracy data not available.',
                            horizontalalignment='center', verticalalignment='center', transform=axes[1, 0].transAxes)


        # Plot 4: Overall confusion matrix
        if len(results['all_true_labels']) > 0:
            cm = confusion_matrix(results['all_true_labels'], results['all_predictions'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 1], cmap='Blues', cbar=True)
            axes[1, 1].set_title('Overall Confusion Matrix', fontsize=14)
            axes[1, 1].set_xlabel('Predicted Label', fontsize=12)
            axes[1, 1].set_ylabel('True Label', fontsize=12)
        else:
            axes[1, 1].set_title('Overall Confusion Matrix (No Data)', fontsize=14)
            axes[1, 1].text(0.5, 0.5, 'Overall confusion matrix data not available.',
                            horizontalalignment='center', verticalalignment='center', transform=axes[1, 1].transAxes)


        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent suptitle overlap

        if save_plots:
            if self.params['sbatch']:
                # do sbatch stuff
                plot_path = self.save_dir / f"evaluation_plots_{self.timestamp}_{self.params['task_id']}.png"
            else:
                plot_path = self.save_dir / f"evaluation_plots_{self.timestamp}.png"

            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {plot_path}")

        plt.show()

    def save_results(self, results):
        """Save evaluation results to pickle file"""
        if self.params['sbatch']:
            # do sbatch stuff
            results_path = self.save_dir / f"test_results_{self.timestamp}_{self.params['task_id']}.pkl"
        else:
            results_path = self.save_dir / f"test_results_{self.timestamp}.pkl"

        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Test results saved to {results_path}")
        return results_path



# Utility functions for standalone use (no changes needed for this request)
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
