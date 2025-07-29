#!./env/bin/python
"""
Compare TA-A-GEM experiments across multiple directories.

Usage:
    python compare_experiments.py dir1 dir2 [dir3 ...]

Example:
    python compare_experiments.py test_results baseline_results improved_results

This script will:
1. Load all pickle files from each directory
2. Perform statistical comparisons (99% confidence)
3. Generate accuracy vs epoch plots with std deviation bands
4. Save results to CSV and HTML plot
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import argparse
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Import functions from retro_stats.py
from retro_stats import load_pickle_files

# Static image export controls
EXPORT_FORMATS = {
    "accuracy_plots": False,  # Accuracy vs epochs plots
    "html": False,  # Save HTML format
    "show_html": False,  # Show HTML image
    "png": False,  # Save PNG format
    "svg": False,  # Save SVG format
    "pdf": False,  # Save PDF format
    "csv": False,  # Save a csv of the data
}

# This import is not needed for this script, but it is needed to export images
if any([EXPORT_FORMATS["png"], EXPORT_FORMATS["pdf"], EXPORT_FORMATS["svg"]]):
    import kaleido


def load_multiple_directories(directories):
    """Load pickle files from multiple directories"""
    all_results = {}

    for directory in directories:
        print(f"Loading results from {directory}...")
        results = load_pickle_files(
            directory=directory, num_files=100
        )  # Load all files
        all_results[directory] = results
        print(f"  Found {len(results)} files")

    return all_results


def extract_epoch_accuracies(results):
    """Extract accuracy per epoch for plotting"""
    epoch_accuracies = []

    for result in results:
        if result["final_accuracy"] is None:
            continue

        data = result["data"]
        task_type = result["task_type"]

        # Extract epoch-by-epoch accuracies
        if "epoch_data" in data and data["epoch_data"]:
            accuracies = [epoch["overall_accuracy"] for epoch in data["epoch_data"]]
        elif "task_accuracies" in data and data["task_accuracies"]:
            accuracies = data["task_accuracies"]
        else:
            continue

        epoch_accuracies.append(
            {
                "task_type": task_type,
                "accuracies": accuracies,
                "num_epochs": len(accuracies),
            }
        )

    return epoch_accuracies


def check_parameter_differences(all_results):
    """Check for parameter differences between directories, excluding task_type"""
    directory_params = {}

    # Collect parameters from each directory
    for directory, results in all_results.items():
        params_set = set()
        for result in results:
            if result["data"] and "params" in result["data"]:
                params = result["data"]["params"].copy()
                # Remove task-dependent parameters that are expected to vary
                # Also remove any params that don't affect the algorithm
                ignore = [
                    "experiment_name",
                    "task_type",
                    "num_tasks",
                    "num_pools",
                    "memory_size_q",
                    "output_dir",
                    "verbose",
                    "tracking_interval"
                ]
                for p in ignore:
                    params.pop(p, None)
                # Convert to frozenset of items for hashability
                params_set.add(frozenset(params.items()))

        # Use the first parameter set as representative
        directory_params[directory] = dict(list(params_set)[0]) if params_set else {}

    # Find differences
    directories = list(directory_params.keys())
    if len(directories) < 2:
        return None

    # Compare each directory against the first one
    baseline_dir = directories[0]
    baseline_params = directory_params[baseline_dir]
    differences = {}

    for other_dir in directories[1:]:
        other_params = directory_params[other_dir]

        # Find parameters that differ
        baseline_diff = {}
        other_diff = {}

        # Check for different values
        all_keys = set(baseline_params.keys()) | set(other_params.keys())
        for key in all_keys:
            baseline_val = baseline_params.get(key)
            other_val = other_params.get(key)

            if baseline_val != other_val:
                baseline_diff[key] = baseline_val
                other_diff[key] = other_val

        if baseline_diff or other_diff:
            differences[f"{baseline_dir}_vs_{other_dir}"] = {
                baseline_dir: baseline_diff,
                other_dir: other_diff
            }

    return differences if differences else None


def perform_statistical_comparison(all_results, task_type_filter=None):
    """Compare results across directories using statistical tests"""
    # Organize data by task type and directory
    task_data = {}

    for directory, results in all_results.items():
        for result in results:
            if result["final_accuracy"] is None:
                continue

            task_type = result["task_type"]

            # Filter by task type if specified
            if task_type_filter and task_type != task_type_filter:
                continue

            if task_type not in task_data:
                task_data[task_type] = {}

            if directory not in task_data[task_type]:
                task_data[task_type][directory] = []

            task_data[task_type][directory].append(result["final_accuracy"])

    # Perform pairwise comparisons
    comparison_results = {}

    for task_type, dir_data in task_data.items():
        if len(dir_data) < 2:
            continue

        directories = list(dir_data.keys())
        comparison_results[task_type] = {
            "directories": directories,
            "means": {},
            "stds": {},
            "counts": {},
            "best_dirs": [],
            "pairwise_tests": {},
        }

        # Calculate basic stats
        for directory in directories:
            accuracies = dir_data[directory]
            comparison_results[task_type]["means"][directory] = np.mean(accuracies)
            comparison_results[task_type]["stds"][directory] = (
                np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0
            )
            comparison_results[task_type]["counts"][directory] = len(accuracies)

        # Find statistically best directories
        max_mean = max(comparison_results[task_type]["means"].values())
        best_candidates = [
            d
            for d in directories
            if comparison_results[task_type]["means"][d] == max_mean
        ]

        # Pairwise t-tests with 99% confidence
        significant_differences = []
        for i, dir1 in enumerate(directories):
            for j, dir2 in enumerate(directories):
                if i >= j:
                    continue

                data1 = dir_data[dir1]
                data2 = dir_data[dir2]

                if len(data1) < 2 or len(data2) < 2:
                    continue

                # Welch's t-test (unequal variances)
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)

                comparison_results[task_type]["pairwise_tests"][f"{dir1}_vs_{dir2}"] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant_99": p_value < 0.01,
                }

                if p_value < 0.01:
                    better_dir = dir1 if np.mean(data1) > np.mean(data2) else dir2
                    significant_differences.append(
                        (better_dir, dir1 if better_dir == dir2 else dir2, p_value)
                    )

        # Determine best directories with statistical confidence
        if significant_differences:
            # Find directories that are significantly better than others
            winners = set()
            for better, worse, p_val in significant_differences:
                winners.add(better)

            # Remove any winner that loses to another winner
            final_winners = winners.copy()
            for winner in winners:
                for other_winner in winners:
                    if winner != other_winner:
                        key1 = f"{winner}_vs_{other_winner}"
                        key2 = f"{other_winner}_vs_{winner}"
                        test_key = (
                            key1
                            if key1 in comparison_results[task_type]["pairwise_tests"]
                            else key2
                        )

                        if test_key in comparison_results[task_type]["pairwise_tests"]:
                            test = comparison_results[task_type]["pairwise_tests"][
                                test_key
                            ]
                            if test["significant_99"]:
                                mean1 = comparison_results[task_type]["means"][winner]
                                mean2 = comparison_results[task_type]["means"][
                                    other_winner
                                ]
                                if mean1 < mean2 and winner in final_winners:
                                    final_winners.remove(winner)

            comparison_results[task_type]["best_dirs"] = list(final_winners)
        else:
            # No significant differences, all are equally good
            comparison_results[task_type]["best_dirs"] = best_candidates

    return comparison_results


def convert_color_to_rgba(color, alpha=0.2):
    """Convert color to rgba format safely"""
    if color.startswith("rgb("):
        rgb_vals = color[4:-1].split(",")
        return f"rgba({rgb_vals[0]},{rgb_vals[1]},{rgb_vals[2]},{alpha})"
    else:
        rgb_tuple = px.colors.hex_to_rgb(color)
        return f"rgba({rgb_tuple[0]},{rgb_tuple[1]},{rgb_tuple[2]},{alpha})"


def pad_accuracy_sequences(accuracy_lists):
    """Pad accuracy sequences to same length and return statistics"""
    max_length = max(len(acc_list) for acc_list in accuracy_lists)

    padded_accuracies = []
    for acc_list in accuracy_lists:
        assert len(acc_list) == max_length, "Cannot compare unequal lengths"
        # if len(acc_list) < max_length:
        # padded = acc_list + [acc_list[-1]] * (max_length - len(acc_list))
        # else:
        padded = acc_list
        padded_accuracies.append(padded)

    accuracy_matrix = np.array(padded_accuracies)
    mean_accuracy = np.mean(accuracy_matrix, axis=0)
    std_accuracy = np.std(accuracy_matrix, axis=0)
    epochs = list(range(len(mean_accuracy)))

    return epochs, mean_accuracy, std_accuracy


def add_accuracy_traces(fig, epochs, mean_accuracy, std_accuracy, directory, color):
    """Add mean line and std deviation band to figure"""
    # Add mean line
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=mean_accuracy,
            mode="lines",
            name=f"{directory}",
            line=dict(color=color),
            legendgroup=f"{directory}",
        )
    )

    # Add std deviation band
    rgba_color = convert_color_to_rgba(color)
    fig.add_trace(
        go.Scatter(
            x=epochs + epochs[::-1],
            y=(mean_accuracy + std_accuracy).tolist()
            + (mean_accuracy - std_accuracy)[::-1].tolist(),
            fill="toself",
            fillcolor=rgba_color,
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            hoverinfo="skip",
            legendgroup=f"{directory}",
        )
    )


def create_task_plot(task_dirs, task_type, colors):
    """Create a single plot for a task type"""
    fig = go.Figure()

    for idx, (directory, accuracy_lists) in enumerate(task_dirs.items()):
        epochs, mean_accuracy, std_accuracy = pad_accuracy_sequences(accuracy_lists)
        color = colors[idx % len(colors)]
        add_accuracy_traces(fig, epochs, mean_accuracy, std_accuracy, directory, color)

    fig.update_layout(
        title=f"Accuracy vs Epochs - {task_type.title()} (Mean ± 1 Standard Deviation)",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        hovermode="x unified",
        yaxis=dict(range=[0, 1]),
        height=700,
    )

    return fig


def save_or_show_figure(
    figure: go.Figure, plot_type: str, task_type: str, output_dir: str | None = None
):
    """Save figure to static images and/or show based on configuration"""
    if output_dir and EXPORT_FORMATS["accuracy_plots"]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{plot_type}_{task_type}_{timestamp}"
        # Save in requested formats
        if EXPORT_FORMATS["png"]:
            png_filename = os.path.join(output_dir, f"{base_filename}.png")
            figure.write_image(png_filename)
            print(f"PNG saved to: {png_filename}")

        if EXPORT_FORMATS["svg"]:
            svg_filename = os.path.join(output_dir, f"{base_filename}.svg")
            figure.write_image(svg_filename)
            print(f"SVG saved to: {svg_filename}")

        if EXPORT_FORMATS["pdf"]:
            pdf_filename = os.path.join(output_dir, f"{base_filename}.pdf")
            figure.write_image(pdf_filename)
            print(f"PDF saved to: {pdf_filename}")

    # Save HTML version
    if output_dir and EXPORT_FORMATS["html"]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_filename = os.path.join(
            output_dir, f"{plot_type}_{task_type}_{timestamp}.html"
        )
        figure.write_html(html_filename)
        print(f"HTML plot saved to: {html_filename}")
    if EXPORT_FORMATS["show_html"]:
        figure.show()


def create_epoch_plot(all_results, task_filter=None):
    """Create plotly graph showing accuracy vs epochs with std deviation bands"""
    colors = px.colors.qualitative.Set1

    # Group all data by task type first
    all_task_data = {}

    for directory, results in all_results.items():
        epoch_data = extract_epoch_accuracies(results)
        if not epoch_data:
            continue

        # Group by task type
        for item in epoch_data:
            task_type = item["task_type"]

            # Filter by task type if specified
            if task_filter and task_type != task_filter:
                continue

            if task_type not in all_task_data:
                all_task_data[task_type] = {}
            if directory not in all_task_data[task_type]:
                all_task_data[task_type][directory] = []
            all_task_data[task_type][directory].append(item["accuracies"])

    # Create plots
    if task_filter:
        # Single plot for specified task type
        if task_filter not in all_task_data:
            print(f"Warning: No data found for task type '{task_filter}'")
            return [go.Figure()]

        fig = create_task_plot(all_task_data[task_filter], task_filter, colors)
        return [fig]
    else:
        # Multiple plots - one for each task type
        figures = []
        for task_type, task_dirs in all_task_data.items():
            fig = create_task_plot(task_dirs, task_type, colors)
            figures.append((task_type, fig))
        return figures


def print_comparison_results(comparison_results):
    """Print statistical comparison results"""
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON RESULTS (99% Confidence)")
    print("=" * 80)

    for task_type, results in comparison_results.items():
        print(f"\n{task_type.upper()}:")
        print("-" * 40)

        # Print basic statistics
        print("Directory Statistics:")
        for directory in results["directories"]:
            mean_acc = results["means"][directory]
            std_acc = results["stds"][directory]
            count = results["counts"][directory]
            print(f"  {directory}: {mean_acc:.4f} ± {std_acc:.4f} (n={count})")

        # Print best directories
        if results["best_dirs"]:
            print(f"Best performing directory(ies): {', '.join(results['best_dirs'])}")
        else:
            print("No statistically significant differences found")

        # Print significant pairwise comparisons
        significant_tests = [
            (k, v) for k, v in results["pairwise_tests"].items() if v["significant_99"]
        ]
        if significant_tests:
            print("Significant pairwise differences (p < 0.01):")
            for test_name, test_result in significant_tests:
                dir1, dir2 = test_name.split("_vs_")
                mean1 = results["means"][dir1]
                mean2 = results["means"][dir2]
                better = dir1 if mean1 > mean2 else dir2
                print(
                    f"  {better} > {dir1 if better == dir2 else dir2} (p = {test_result['p_value']:.6f})"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Compare TA-A-GEM Experiments Across Directories",
        epilog="""
Examples:
  %(prog)s test_results baseline_results
  %(prog)s experiment1 experiment2 experiment3 --output_dir results
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "directories",
        nargs="+",
        help="Two or more directories containing pickle files to compare",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save comparison results (default: None)",
    )
    parser.add_argument(
        "--task_type",
        "-t",
        type=str,
        choices=["permutation", "rotation", "class_split"],
        help="Filter results to specific task type only",
    )
    args = parser.parse_args()

    # if len(args.directories) < 2:
    #     print("Error: Please provide at least two directories to compare")
    #     return

    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    print("Multi-Directory TA-A-GEM Experiment Comparison")
    print("=" * 60)
    print(f"Comparing directories: {', '.join(args.directories)}")

    # Load all results
    all_results = load_multiple_directories(args.directories)

    # Check for parameter differences
    param_differences = check_parameter_differences(all_results)
    if param_differences:
        print("\nWARNING: Parameter differences detected between directories:")
        print("=" * 60)
        for comparison_key, dirs_diff in param_differences.items():
            print(f"\nParameter mismatches found:")
            for directory, params in dirs_diff.items():
                print(f"{directory}:")
                for key, value in params.items():
                    print(f"- {key}: {value}")
        print("\n" + "=" * 60)

    # Perform statistical comparison
    comparison_results = perform_statistical_comparison(all_results, args.task_type)

    # Print results
    print_comparison_results(comparison_results)

    # Create and save plot(s)
    figures = create_epoch_plot(all_results, args.task_type)

    if args.task_type:
        # Single plot for specific task type
        save_or_show_figure(
            figures[0], "accuracy_comparison", args.task_type, args.output_dir
        )
    else:
        # Multiple plots - one for each task type
        for task_type, fig in figures:
            save_or_show_figure(fig, "accuracy_comparison", task_type, args.output_dir)

    # Save statistical results to CSV
    if args.output_dir and EXPORT_FORMATS["csv"]:
        summary_data = []
        for task_type, results in comparison_results.items():
            for directory in results["directories"]:
                summary_data.append(
                    {
                        "Task Type": task_type,
                        "Directory": directory,
                        "Mean Accuracy": results["means"][directory],
                        "Std Deviation": results["stds"][directory],
                        "Count": results["counts"][directory],
                        "Is Best": directory in results["best_dirs"],
                    }
                )

        summary_df = pd.DataFrame(summary_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = os.path.join(
            args.output_dir, f"comparison_summary_{timestamp}.csv"
        )
        summary_df.to_csv(summary_filename, index=False)
        print(f"Summary results saved to: {summary_filename}")

    print("\nComparison complete!")


if __name__ == "__main__":
    main()
