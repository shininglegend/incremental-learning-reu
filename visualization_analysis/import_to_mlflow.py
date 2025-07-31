"""
Import existing experiment results into MLflow.

This script loads pickle files from previous experiments and imports them
into MLflow for unified tracking and comparison.

Usage:
    python utils/import_to_mlflow.py --input_dir test_results/my-experiment --last 5
    python utils/import_to_mlflow.py --input_dir test_results/my-experiment --all

NOTE: References to epochs in this file (and others) frequently are 
references to the tests run every TESTING_INTERVAL batches.
The code is kept as-is to allow it to read legacy runs.
"""

import argparse
import os
import pickle
import glob
from datetime import datetime
import re

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available. Install with: pip install mlflow")
    exit(1)


def load_pickle_data(filepath):
    """Load and process pickle data, converting to legacy format if needed."""
    try:
        with open(filepath, 'rb') as file:
            data = pickle.load(file)

        # Convert new format to legacy format if needed
        if "epoch_data" in data and data["epoch_data"]:
            epoch_data = data["epoch_data"]

            # Convert to legacy format for compatibility
            data["task_accuracies"] = [ep["overall_accuracy"] for ep in epoch_data]
            data["per_task_accuracies"] = [ep["individual_accuracies"] for ep in epoch_data]
            data["memory_sizes"] = [ep["memory_size"] for ep in epoch_data]
            data["epoch_losses"] = [[ep["epoch_loss"]] for ep in epoch_data]
            data["memory_efficiency"] = [
                ep["overall_accuracy"] / ep["memory_size"] if ep["memory_size"] > 0 else 0.0
                for ep in epoch_data
            ]

            # Group by task for epoch losses
            task_epoch_losses = {}
            for ep in epoch_data:
                task_id = ep["task_id"]
                if task_id not in task_epoch_losses:
                    task_epoch_losses[task_id] = []
                task_epoch_losses[task_id].append(ep["epoch_loss"])

            data["epoch_losses"] = [
                task_epoch_losses[i] for i in sorted(task_epoch_losses.keys())
            ]

        return data

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def import_run_to_mlflow(filepath, experiment_name):
    """Import a single pickle file as an MLflow run."""
    data = load_pickle_data(filepath)
    if not data:
        return False

    try:
        # Extract run information from filename and data
        filename = os.path.basename(filepath)
        timestamp_obj = os.path.getmtime(filepath)

        # Extract task type from filename or params
        task_type = "unknown"
        if 'params' in data and 'task_type' in data['params']:
            task_type = data['params']['task_type']
        elif any(t in filename for t in ['perm', 'rot', 'cla']):
            if 'perm' in filename:
                task_type = 'permutation'
            elif 'rot' in filename:
                task_type = 'rotation'
            elif 'cla' in filename:
                task_type = 'class_split'

        run_name = f"{task_type}_{timestamp_obj}"

        # Start MLflow run
        with mlflow.start_run(run_name=run_name) as run:
            print(f"Importing {filename} as run: {run_name}")

            # Log parameters
            if 'params' in data:
                # Clean parameters for MLflow (must be strings)
                clean_params = {}
                for key, value in data['params'].items():
                    clean_params[key] = str(value)
                mlflow.log_params(clean_params)

            # Log metrics from epoch_data if available
            if 'epoch_data' in data and data['epoch_data']:
                for step, epoch in enumerate(data['epoch_data']):
                    metrics = {
                        'overall_accuracy': epoch['overall_accuracy'],
                        'epoch_loss': epoch['epoch_loss'],
                        'memory_size': epoch['memory_size'],
                        'current_task': epoch['task_id'],
                    }

                    if epoch.get('learning_rate') is not None:
                        metrics['learning_rate'] = epoch['learning_rate']
                    if epoch.get('training_time') is not None:
                        metrics['training_time'] = epoch['training_time']

                    # Log individual task accuracies
                    for i, acc in enumerate(epoch['individual_accuracies']):
                        metrics[f'task_{i}_accuracy'] = acc

                    mlflow.log_metrics(metrics, step=step)

            # Log legacy format metrics if epoch_data not available
            elif 'task_accuracies' in data:
                for step, accuracy in enumerate(data['task_accuracies']):
                    metrics = {'overall_accuracy': accuracy}

                    if 'memory_sizes' in data and step < len(data['memory_sizes']):
                        metrics['memory_size'] = data['memory_sizes'][step]

                    if 'memory_efficiency' in data and step < len(data['memory_efficiency']):
                        metrics['memory_efficiency'] = data['memory_efficiency'][step]

                    mlflow.log_metrics(metrics, step=step)

            # Log batch losses (sampled to avoid overwhelming MLflow)
            if 'batch_losses' in data and data['batch_losses']:
                for i, batch_loss in enumerate(data['batch_losses'][::10]):  # Sample every 10th
                    mlflow.log_metric("batch_loss", batch_loss['loss'], step=i)
                    mlflow.log_metric("batch_task", batch_loss['task'], step=i)

            # Log summary metrics
            if 'task_accuracies' in data and data['task_accuracies']:
                final_accuracy = data['task_accuracies'][-1]
                avg_accuracy = sum(data['task_accuracies']) / len(data['task_accuracies'])

                summary_metrics = {
                    'final_overall_accuracy': final_accuracy,
                    'average_accuracy_all_epochs': avg_accuracy,
                    'total_epochs': len(data['task_accuracies']),
                }

                if 'memory_sizes' in data and data['memory_sizes']:
                    summary_metrics['final_memory_size'] = data['memory_sizes'][-1]

                if 'params' in data and 'num_tasks' in data['params']:
                    summary_metrics['total_tasks'] = data['params']['num_tasks']

                mlflow.log_metrics(summary_metrics)

            # Log the original pickle file as an artifact
            mlflow.log_artifact(filepath, "original_data")

            # Add tags for easier filtering
            tags = {
                'source': 'imported',
                'original_filename': filename,
                'import_timestamp': datetime.now().isoformat(),
            }

            if 'params' in data:
                if 'task_type' in data['params']:
                    tags['task_type'] = data['params']['task_type']
                if 'dataset_name' in data['params']:
                    tags['dataset'] = data['params']['dataset_name']
                if 'quick_test_mode' in data['params']:
                    tags['quick_test'] = str(data['params']['quick_test_mode'])

            mlflow.set_tags(tags)

        print(f"Successfully imported: {filename}")
        return True

    except Exception as e:
        print(f"Error importing {filepath}: {e}")
        return False


def find_pickle_files(input_dir):
    """Find all pickle files in the input directory."""
    pickle_files = glob.glob(os.path.join(input_dir, "*.pkl"))

    # Sort files by modification time (newest first) and take only the specified number
    pickle_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return pickle_files


def main():
    parser = argparse.ArgumentParser(description="Import pickle files to MLflow")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing pickle files to import"
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        help="Import only the last N files (sorted by timestamp)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Import all pickle files found"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be imported without actually importing"
    )

    args = parser.parse_args()

    if not MLFLOW_AVAILABLE:
        print("MLflow is required for this script")
        return 1

    if not os.path.exists(args.input_dir):
        print(f"Input directory does not exist: {args.input_dir}")
        return 1

    if not args.all and args.last is None:
        print("Must specify either --all or --last N")
        return 1

    # Extract experiment name from directory name
    experiment_name = os.path.basename(args.input_dir.rstrip('/'))

    # Find pickle files
    pickle_files = find_pickle_files(args.input_dir)

    if not pickle_files:
        print(f"No pickle files found in {args.input_dir}")
        return 1

    # Filter files if --last specified
    if args.last is not None:
        pickle_files = pickle_files[:args.last]

    print(f"Found {len(pickle_files)} pickle files to import:")
    for f in pickle_files:
        print(os.path.basename(f))

    if args.dry_run:
        print("\nDry run - no files imported")
        return 0

    # Set up MLflow experiment
    try:
        mlflow.set_experiment(experiment_name)
        print(f"\nImporting to MLflow experiment: {experiment_name}")
    except Exception as e:
        print(f"Failed to setup MLflow experiment: {e}")
        return 1

    # Import files
    success_count = 0
    for filepath in pickle_files:
        if import_run_to_mlflow(filepath, experiment_name):
            success_count += 1

    print(f"\nImport complete: {success_count}/{len(pickle_files)} files imported successfully")
    print(f"View results with: mlflow ui")

    return 0


if __name__ == "__main__":
    exit(main())
