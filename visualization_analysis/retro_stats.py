import pickle
import pandas as pd
import numpy as np
from scipy import stats
import os
import glob
import argparse
from datetime import datetime


def load_pickle_files(directory, num_files=15):
    """Load pickle files from the test_results directory

    NOTE: Used by compare_experiments.py - do not modify signature
    """
    pickle_files = glob.glob(os.path.join(directory, "*.pkl"))

    # Sort files by modification time (newest first) and take only the specified number
    pickle_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    pickle_files = pickle_files[:num_files]

    results = []
    for file_path in pickle_files:
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            # Extract task type from params if available
            task_type = "unknown"
            if "params" in data and "task_type" in data["params"]:
                task_type = data["params"]["task_type"]

            # Extract final accuracy - handle both new and legacy formats
            final_accuracy = None
            # Extract average first-task accuracy across all epochs
            avg_first_task_accuracy = None
            if "epoch_data" in data and data["epoch_data"]:
                final_accuracy = sum(
                    ep["overall_accuracy"] for ep in data["epoch_data"]
                ) / len(data["epoch_data"])
                avg_first_task_accuracy = sum(
                    ep["individual_accuracies"][0] for ep in data["epoch_data"]
                ) / len(data["epoch_data"])
            elif "per_task_accuracies" in data and data["per_task_accuracies"]:
                print("Warning: Old format detected.")
                # Legacy format: average accuracy across all evaluations and tasks
                all_accuracies_by_overall = []
                for task_eval in data["per_task_accuracies"]:
                    if task_eval:
                        all_accuracies_by_overall.extend(task_eval)
                if all_accuracies_by_overall:
                    final_accuracy = sum(all_accuracies_by_overall) / len(
                        all_accuracies_by_overall
                    )

            # Extract timestamp
            timestamp = data.get("timestamp", "unknown")

            results.append(
                {
                    "file": os.path.basename(file_path),
                    "task_type": task_type,
                    "final_accuracy": final_accuracy,
                    "avg_first_task_accuracy": avg_first_task_accuracy,
                    "timestamp": timestamp,
                    "data": data,
                    "file_mtime": os.path.getmtime(file_path),
                }
            )

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return results


def compute_confidence_interval(data, confidence=0.99):
    """Compute confidence interval for given data

    NOTE: Used by compare_experiments.py - do not modify signature
    """
    n = len(data)
    if n < 2:
        return np.nan, np.nan

    mean = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2.0, n - 1)

    return mean - h, mean + h


def analyze_experiments(results):
    """Analyze experiments and compute statistics

    NOTE: Used by compare_experiments.py - do not modify signature
    """

    # Group results by task type
    task_groups = {}
    forgetting_groups = {}
    first_task_groups = {}

    for result in results:
        task_type = result["task_type"]
        if task_type not in task_groups:
            task_groups[task_type] = []
            forgetting_groups[task_type] = []
            first_task_groups[task_type] = []

        if result["final_accuracy"] is not None:
            task_groups[task_type].append(result["final_accuracy"])

            # Calculate forgetting for first task
            forgetting = calculate_first_task_forgetting(result["data"])
            if forgetting is not None:
                forgetting_groups[task_type].append(forgetting)

        # Track average first-task accuracy
        if result["avg_first_task_accuracy"] is not None:
            first_task_groups[task_type].append(result["avg_first_task_accuracy"])

    # Compute statistics for each task type
    statistics = []

    for task_type, accuracies in task_groups.items():
        if len(accuracies) == 0:
            continue

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0

        # Compute 99% confidence interval
        ci_lower, ci_upper = compute_confidence_interval(accuracies, confidence=0.99)

        # Forgetting statistics
        forgetting_values = forgetting_groups[task_type]
        mean_forgetting = np.mean(forgetting_values) if forgetting_values else np.nan
        std_forgetting = (
            np.std(forgetting_values, ddof=1) if len(forgetting_values) > 1 else 0
        )
        forgetting_ci_lower, forgetting_ci_upper = (
            compute_confidence_interval(forgetting_values, confidence=0.99)
            if forgetting_values
            else (np.nan, np.nan)
        )

        # First-task accuracy statistics
        first_task_values = first_task_groups[task_type]
        mean_first_task = np.mean(first_task_values) if first_task_values else np.nan
        std_first_task = (
            np.std(first_task_values, ddof=1) if len(first_task_values) > 1 else 0
        )
        first_task_ci_lower, first_task_ci_upper = (
            compute_confidence_interval(first_task_values, confidence=0.99)
            if first_task_values
            else (np.nan, np.nan)
        )

        statistics.append(
            {
                "Task Type": task_type,
                "Number of Runs": len(accuracies),
                "Mean Accuracy": mean_accuracy,
                "Std Deviation": std_accuracy,
                "99% CI Lower": ci_lower,
                "99% CI Upper": ci_upper,
                "Raw Accuracies": accuracies,
                "Mean Forgetting": mean_forgetting,
                "Forgetting Std": std_forgetting,
                "Forgetting CI Lower": forgetting_ci_lower,
                "Forgetting CI Upper": forgetting_ci_upper,
                "Raw Forgetting": forgetting_values,
                "Mean First Task": mean_first_task,
                "First Task Std": std_first_task,
                "First Task CI Lower": first_task_ci_lower,
                "First Task CI Upper": first_task_ci_upper,
                "Raw First Task": first_task_values,
            }
        )

    return statistics


def calculate_first_task_forgetting(data):
    """Calculate average forgetting for the first task across all epochs after first task completion"""
    assert data["epoch_data"], "epoch_data is empty"

    epoch_data = data["epoch_data"]
    max_first_task_accuracy = 0.0
    forgetting_values = []

    for epoch_info in epoch_data:
        assert (
            len(epoch_info["individual_accuracies"]) > 0
        ), f"individual_accuracies is empty for epoch {epoch_info}"

        first_task_accuracy = epoch_info["individual_accuracies"][0]

        # Update max accuracy seen so far for first task until the first task is complete
        if epoch_info["task_id"] == 0:
            max_first_task_accuracy = max(max_first_task_accuracy, first_task_accuracy)
            continue

        # After first task training is complete, calculate forgetting for this epoch
        forgetting = max_first_task_accuracy - first_task_accuracy
        forgetting_values.append(forgetting)

    if len(forgetting_values) == 0:
        return None

    return sum(forgetting_values) / len(forgetting_values)


def create_summary_table(statistics):
    """Create formatted summary table"""

    # Create DataFrame
    df = pd.DataFrame(statistics)

    # Remove raw data from display version
    display_df = df.drop(["Raw Accuracies", "Raw Forgetting", "Raw First Task"], axis=1)

    # Format numerical columns
    display_df["Mean Accuracy"] = display_df["Mean Accuracy"].apply(
        lambda x: f"{x:.4f}"
    )
    display_df["Std Deviation"] = display_df["Std Deviation"].apply(
        lambda x: f"{x:.4f}"
    )
    display_df["99% CI Lower"] = display_df["99% CI Lower"].apply(
        lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
    )
    display_df["99% CI Upper"] = display_df["99% CI Upper"].apply(
        lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
    )
    display_df["Mean Forgetting"] = display_df["Mean Forgetting"].apply(
        lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
    )
    display_df["Forgetting Std"] = display_df["Forgetting Std"].apply(
        lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
    )
    display_df["Forgetting CI Lower"] = display_df["Forgetting CI Lower"].apply(
        lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
    )
    display_df["Forgetting CI Upper"] = display_df["Forgetting CI Upper"].apply(
        lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
    )
    display_df["Mean First Task"] = display_df["Mean First Task"].apply(
        lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
    )
    display_df["First Task Std"] = display_df["First Task Std"].apply(
        lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
    )
    display_df["First Task CI Lower"] = display_df["First Task CI Lower"].apply(
        lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
    )
    display_df["First Task CI Upper"] = display_df["First Task CI Upper"].apply(
        lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
    )

    return display_df, df


def main():
    parser = argparse.ArgumentParser(description="TA-A-GEM Experiment Analysis")
    parser.add_argument(
        "-n",
        "--num_runs",
        type=int,
        default=15,
        help="Number of most recent test runs to analyze (default: 15)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="test_results",
        help="Directory containing the test results (default: test_results)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the analysis results (default: test_results)",
    )
    args = parser.parse_args()

    print("TA-A-GEM Experiment Analysis")
    print("=" * 50)

    # Load all pickle files
    results = load_pickle_files(directory=args.input_dir, num_files=args.num_runs)

    if not results:
        print(f"No pickle files found in {args.input_dir} directory")
        return

    print(
        f"Found {len(results)} result files (analyzing last {args.num_runs} runs only)"
    )

    # Show time range of analyzed files
    mtimes = [result["file_mtime"] for result in results]
    earliest_time = min(mtimes)
    latest_time = max(mtimes)

    earliest_str = datetime.fromtimestamp(earliest_time).strftime(
        "%B %d, %Y, at %I:%M:%S %p"
    )
    latest_str = datetime.fromtimestamp(latest_time).strftime(
        "%B %d, %Y, at %I:%M:%S %p"
    )

    print(f"Test results are from {earliest_str} to {latest_str}")

    # Show breakdown by task type
    task_counts = {}
    for result in results:
        task_type = result["task_type"]
        task_counts[task_type] = task_counts.get(task_type, 0) + 1

    print("\nBreakdown by task type:")
    for task_type, count in task_counts.items():
        print(f"  {task_type}: {count} runs")

    # Analyze experiments
    statistics = analyze_experiments(results)

    if not statistics:
        print("No valid experiment results found")
        return

    # Sort for consistent viewing
    statistics = sorted(statistics, key=lambda a: a["Task Type"])

    # Create summary table
    display_df, full_df = create_summary_table(statistics)

    # Print results to terminal
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print(display_df.to_string(index=False))

    # Print detailed raw accuracies
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    to_copy_accuracies = ""
    to_copy_forgetting = ""
    to_copy_first_task = ""
    for stat in statistics:
        print(f"\n{stat['Task Type']} ({stat['Number of Runs']} runs):")
        print(
            f"  Raw accuracies: [{', '.join(f'{acc:.4f}' for acc in stat['Raw Accuracies'])}]"
        )
        to_copy_accuracies += (
            ",".join(f"{acc:.4f}" for acc in stat["Raw Accuracies"]) + "\n"
        )
        print(f"  Mean: {stat['Mean Accuracy']:.4f} ± {stat['Std Deviation']:.4f}")
        print(f"  99% CI: [{stat['99% CI Lower']:.4f}, {stat['99% CI Upper']:.4f}]")

        # Forgetting statistics
        if stat["Raw Forgetting"]:
            print("  --")
            print(
                f"  Raw forgetting: [{', '.join(f'{fg:.4f}' for fg in stat['Raw Forgetting'])}]"
            )
            to_copy_forgetting += (
                ",".join(f"{fg:.4f}" for fg in stat["Raw Forgetting"]) + "\n"
            )
            print(
                f"  Forgetting Mean: {stat['Mean Forgetting']:.4f} ± {stat['Forgetting Std']:.4f}"
            )
            print(
                f"  Forgetting 99% CI: [{stat['Forgetting CI Lower']:.4f}, {stat['Forgetting CI Upper']:.4f}]"
            )
        else:
            print("  No forgetting data available")

        # First-task accuracy statistics
        if stat["Raw First Task"]:
            print("  --")
            print(
                f"  Raw first task: [{', '.join(f'{ft:.4f}' for ft in stat['Raw First Task'])}]"
            )
            to_copy_first_task += (
                ",".join(f"{ft:.4f}" for ft in stat["Raw First Task"]) + "\n"
            )
            print(
                f"  First Task Mean: {stat['Mean First Task']:.4f} ± {stat['First Task Std']:.4f}"
            )
            print(
                f"  First Task 99% CI: [{stat['First Task CI Lower']:.4f}, {stat['First Task CI Upper']:.4f}]"
            )
        else:
            print("  No first task data available")

    print(
        f"\nTo copy:\n-- Accuracies\n{to_copy_accuracies}-- Forgetting\n{to_copy_forgetting}-- First Task\n{to_copy_first_task}"
    )
    if stat["Raw First Task"]:
        print(
            "Note: First Task accuracy is calculated over all epochs after the first 20."
        )

    # Save results to CSV
    if args.output_dir is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save summary table
        summary_filename = os.path.join(
            args.output_dir, f"experiment_analysis_{timestamp}.csv"
        )
        display_df.to_csv(summary_filename, index=False)
        print(f"\nSummary results saved to: {summary_filename}")

        # Save detailed results
        detailed_results = []
        for stat in statistics:
            for i, accuracy in enumerate(stat["Raw Accuracies"]):
                detailed_results.append(
                    {
                        "Task Type": stat["Task Type"],
                        "Run Number": i + 1,
                        "Final Accuracy": accuracy,
                    }
                )

        detailed_df = pd.DataFrame(detailed_results)
        detailed_filename = os.path.join(
            args.output_dir, f"detailed_results_{timestamp}.csv"
        )
        detailed_df.to_csv(detailed_filename, index=False)
        print(f"Detailed results saved to: {detailed_filename}")

    print("Analysis complete!")


if __name__ == "__main__":
    main()
