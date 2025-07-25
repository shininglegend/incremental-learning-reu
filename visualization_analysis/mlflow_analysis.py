#!/usr/bin/env python3
"""
MLflow-based experiment comparison tool.

This module provides functionality to query MLflow experiments and generate
comparative visualizations using the plotting functions from compare_experiments.py.

Usage:
    python visualization_analysis/mlflow.py --experiments exp1 exp2 --metric accuracy
    python visualization_analysis/mlflow.py --experiments exp1 exp2 exp3 --task_type permutation
"""

import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

import mlflow
import mlflow.tracking
import plotly.graph_objects as go
import plotly.express as px

from compare_experiments import (
    perform_statistical_comparison,
    create_epoch_plot,
    print_comparison_results,
    convert_color_to_rgba,
    pad_accuracy_sequences,
    add_accuracy_traces,
    create_task_plot
)

def get_mlflow_experiments(experiment_names: Optional[List[str]] = None) -> List[Dict]:
    """Get MLflow experiments by name or all experiments if None."""
    client = mlflow.tracking.MlflowClient()

    if experiment_names:
        experiments = []
        for name in experiment_names:
            try:
                exp = client.get_experiment_by_name(name)
                if exp:
                    experiments.append(exp)
                else:
                    print(f"Warning: Experiment '{name}' not found")
            except Exception as e:
                print(f"Error getting experiment '{name}': {e}")
        return experiments
    else:
        return client.search_experiments()


def get_runs_from_experiment(experiment_id: str, task_filter: Optional[str] = None) -> List[Dict]:
    """Get all runs from an MLflow experiment with optional task filtering."""

    client = mlflow.tracking.MlflowClient()

    # Build filter string
    filter_parts = ["attribute.status = 'FINISHED'"]
    if task_filter:
        filter_parts.append(f"tags.task_type = '{task_filter}'")

    filter_string = " and ".join(filter_parts) if len(filter_parts) > 1 else filter_parts[0]

    try:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            max_results=1000
        )
        return runs
    except Exception as e:
        print(f"Error searching runs: {e}")
        return []


def extract_metrics_from_run(run) -> Dict[str, Any]:
    """Extract relevant metrics from an MLflow run."""
    metrics = run.data.metrics
    params = run.data.params
    tags = run.data.tags

    # Get final accuracy
    final_accuracy = metrics.get('final_overall_accuracy', metrics.get('overall_accuracy'))

    # Get task type from tags or params
    task_type = tags.get('task_type', params.get('task_type', 'unknown'))

    # Get run metadata
    run_info = {
        'run_id': run.info.run_id,
        'run_name': run.info.run_name,
        'start_time': run.info.start_time,
        'end_time': run.info.end_time,
        'status': run.info.status,
        'final_accuracy': final_accuracy,
        'task_type': task_type,
        'params': params,
        'tags': tags,
        'all_metrics': metrics
    }

    return run_info


def get_metric_history(run_id: str, metric_name: str) -> List[Tuple[int, float]]:
    """Get the history of a specific metric for a run."""
    client = mlflow.tracking.MlflowClient()

    try:
        metric_history = client.get_metric_history(run_id, metric_name)
        return [(m.step, m.value) for m in metric_history]
    except Exception as e:
        print(f"Error getting metric history for {metric_name}: {e}")
        return []


def extract_epoch_accuracies_from_mlflow(runs: List[Dict]) -> List[Dict]:
    """Extract epoch-by-epoch accuracies from MLflow runs."""
    epoch_accuracies = []

    for run_info in runs:
        if run_info['final_accuracy'] is None:
            continue

        # Get accuracy history
        accuracy_history = get_metric_history(run_info['run_id'], 'overall_accuracy')

        if not accuracy_history:
            continue

        # Sort by step and extract values
        accuracy_history.sort(key=lambda x: x[0])
        accuracies = [acc for step, acc in accuracy_history]

        epoch_accuracies.append({
            'task_type': run_info['task_type'],
            'accuracies': accuracies,
            'num_epochs': len(accuracies),
            'run_info': run_info
        })

    return epoch_accuracies


def convert_mlflow_data_to_comparison_format(experiment_data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """Convert MLflow data to the format expected by compare_experiments functions."""
    converted_data = {}

    for experiment_name, runs in experiment_data.items():
        converted_runs = []

        for run_info in runs:
            # Create a structure similar to what load_pickle_files returns
            converted_run = {
                'final_accuracy': run_info['final_accuracy'],
                'task_type': run_info['task_type'],
                'data': {
                    'params': run_info['params'],
                    'task_accuracies': None,  # Will be filled if needed
                    'epoch_data': None  # Will be filled if needed
                },
                'filename': run_info['run_name'],
                'timestamp': run_info['start_time']
            }

            # Try to get epoch accuracy data
            accuracy_history = get_metric_history(run_info['run_id'], 'overall_accuracy')
            if accuracy_history:
                accuracy_history.sort(key=lambda x: x[0])
                accuracies = [acc for step, acc in accuracy_history]
                converted_run['data']['task_accuracies'] = accuracies

            converted_runs.append(converted_run)

        converted_data[experiment_name] = converted_runs

    return converted_data


def query_mlflow_experiments(experiment_names: List[str], task_filter: Optional[str] = None) -> Dict[str, List[Dict]]:
    """Query MLflow for experiment data."""

    all_experiment_data = {}

    # Get experiments
    experiments = get_mlflow_experiments(experiment_names)

    for experiment in experiments:
        experiment_name = experiment.name
        print(f"Loading runs from experiment: {experiment_name}")

        # Get runs from this experiment
        runs = get_runs_from_experiment(experiment.experiment_id, task_filter)

        # Extract metrics from runs
        run_data = []
        for run in runs:
            run_info = extract_metrics_from_run(run)
            if run_info['final_accuracy'] is not None:
                run_data.append(run_info)

        all_experiment_data[experiment_name] = run_data
        print(f"  Found {len(run_data)} completed runs")

    return all_experiment_data


def create_mlflow_comparison_plots(experiment_data: Dict[str, List[Dict]], task_filter: Optional[str] = None):
    """Create comparison plots using MLflow data and compare_experiments functions."""


    # Convert MLflow data to comparison format
    comparison_data = convert_mlflow_data_to_comparison_format(experiment_data)

    # Extract epoch accuracies for plotting
    all_task_data = {}
    colors = px.colors.qualitative.Set1

    for experiment_name, runs in comparison_data.items():
        for run in runs:
            if run['final_accuracy'] is None:
                continue

            task_type = run['task_type']

            # Filter by task type if specified
            if task_filter and task_type != task_filter:
                continue

            # Get accuracies
            accuracies = run['data'].get('task_accuracies')
            if not accuracies:
                continue

            if task_type not in all_task_data:
                all_task_data[task_type] = {}
            if experiment_name not in all_task_data[task_type]:
                all_task_data[task_type][experiment_name] = []

            all_task_data[task_type][experiment_name].append(accuracies)

    # Create plots
    figures = []

    if task_filter:
        # Single plot for specified task type
        if task_filter in all_task_data:
            fig = create_task_plot(all_task_data[task_filter], task_filter, colors)
            figures.append((task_filter, fig))
        else:
            print(f"Warning: No data found for task type '{task_filter}'")
    else:
        # Multiple plots - one for each task type
        for task_type, task_dirs in all_task_data.items():
            fig = create_task_plot(task_dirs, task_type, colors)
            figures.append((task_type, fig))

    return figures


def perform_mlflow_statistical_comparison(experiment_data: Dict[str, List[Dict]], task_filter: Optional[str] = None):
    """Perform statistical comparison using MLflow data."""

    # Convert to comparison format
    comparison_data = convert_mlflow_data_to_comparison_format(experiment_data)

    # Use the existing statistical comparison function
    return perform_statistical_comparison(comparison_data, task_filter)


def main():
    parser = argparse.ArgumentParser(
        description="Compare experiments using MLflow data",
        epilog="""
Examples:
  %(prog)s --experiments exp1 exp2 --metric overall_accuracy
  %(prog)s --experiments baseline improved --task_type permutation --output_dir results
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiments", "-e",
        nargs="+",
        required=True,
        help="Names of MLflow experiments to compare"
    )
    parser.add_argument(
        "--task_type", "-t",
        type=str,
        choices=["permutation", "rotation", "class_split"],
        help="Filter results to specific task type only"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save comparison results (default: show plots)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="overall_accuracy",
        help="Primary metric to compare (default: overall_accuracy)"
    )

    args = parser.parse_args()

    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    print("MLflow Experiment Comparison")
    print("=" * 50)
    print(f"Comparing experiments: {', '.join(args.experiments)}")
    print(f"Primary metric: {args.metric}")
    if args.task_type:
        print(f"Task type filter: {args.task_type}")

    # Query MLflow for experiment data
    try:
        experiment_data = query_mlflow_experiments(args.experiments, args.task_type)
    except Exception as e:
        print(f"Error querying MLflow: {e}")
        return 1

    if not experiment_data:
        print("No experiment data found")
        return 1

    # Perform statistical comparison
    comparison_results = perform_mlflow_statistical_comparison(experiment_data, args.task_type)

    # Print results
    if comparison_results:
        print_comparison_results(comparison_results)

    # Create plots
    figures = create_mlflow_comparison_plots(experiment_data, args.task_type)

    # Save or show plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for task_type, fig in figures:
        if args.output_dir:
            plot_filename = os.path.join(
                args.output_dir,
                f"mlflow_comparison_{task_type}_{timestamp}.html"
            )
            fig.write_html(plot_filename)
            print(f"Plot for {task_type} saved to: {plot_filename}")
        else:
            fig.show()

    # Save summary data
    if args.output_dir and comparison_results:
        summary_data = []
        for task_type, results in comparison_results.items():
            for experiment in results["directories"]:
                summary_data.append({
                    "Task Type": task_type,
                    "Experiment": experiment,
                    "Mean Accuracy": results["means"][experiment],
                    "Std Deviation": results["stds"][experiment],
                    "Count": results["counts"][experiment],
                    "Is Best": experiment in results["best_dirs"],
                })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_filename = os.path.join(
                args.output_dir, f"mlflow_comparison_summary_{timestamp}.csv"
            )
            summary_df.to_csv(summary_filename, index=False)
            print(f"Summary results saved to: {summary_filename}")

    print("\nComparison complete!")
    return 0


if __name__ == "__main__":
    exit(main())
