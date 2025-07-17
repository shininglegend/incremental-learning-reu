import pickle
import pandas as pd
import numpy as np
from scipy import stats
import os
import glob
import argparse
from datetime import datetime


def debug_directory(directory):
    """Debug function to check directory and files"""
    print(f"DEBUG: Checking directory: {directory}")
    print(f"DEBUG: Directory exists: {os.path.exists(directory)}")
    print(f"DEBUG: Directory is directory: {os.path.isdir(directory)}")

    if os.path.exists(directory):
        print(f"DEBUG: Directory contents:")
        try:
            contents = os.listdir(directory)
            for item in contents:
                full_path = os.path.join(directory, item)
                print(f"  {item} (file: {os.path.isfile(full_path)})")
        except Exception as e:
            print(f"DEBUG: Error listing directory: {e}")

    # Check glob pattern
    pattern = os.path.join(directory, "ta_agem_metrics_*.pkl")
    print(f"DEBUG: Glob pattern: {pattern}")
    pickle_files = glob.glob(pattern)
    print(f"DEBUG: Found {len(pickle_files)} pickle files matching pattern")
    for f in pickle_files:
        print(f"  {f}")

    # Check for any .pkl files
    all_pkl_pattern = os.path.join(directory, "*.pkl")
    print(f"DEBUG: All .pkl files pattern: {all_pkl_pattern}")
    all_pkl_files = glob.glob(all_pkl_pattern)
    print(f"DEBUG: Found {len(all_pkl_files)} .pkl files total")
    for f in all_pkl_files:
        print(f"  {f}")


def load_pickle_files(directory="test_results", num_files=15):
    """Load pickle files from the test_results directory"""

    debug_directory(directory)


    pickle_files = glob.glob(os.path.join(directory, "ta_agem_metrics_*.pkl"))

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

            # Extract dataset_name - check both direct key and params
            dataset_name = "unknown"
            if "dataset_name" in data:
                dataset_name = data["dataset_name"]
            elif "params" in data and "dataset_name" in data["params"]:
                dataset_name = data["params"]["dataset_name"]

            # Extract removal - check both direct key and params
            removal = "unknown"
            if "removal" in data:
                removal = data["removal"]
            elif "params" in data and "removal" in data["params"]:
                removal = data["params"]["removal"]

            # Extract final accuracy - handle both new and legacy formats
            final_accuracy = None
            if "epoch_data" in data and data["epoch_data"]:
                # New format: average accuracy across all epochs and tasks
                all_accuracies = []
                for epoch in data["epoch_data"]:
                    if epoch["individual_accuracies"]:
                        all_accuracies.extend(epoch["individual_accuracies"])
                if all_accuracies:
                    final_accuracy = sum(all_accuracies) / len(all_accuracies)
            elif "per_task_accuracies" in data and data["per_task_accuracies"]:
                # Legacy format: average accuracy across all evaluations and tasks
                all_accuracies = []
                for task_eval in data["per_task_accuracies"]:
                    if task_eval:
                        all_accuracies.extend(task_eval)
                if all_accuracies:
                    final_accuracy = sum(all_accuracies) / len(all_accuracies)

            # Extract timestamp
            timestamp = data.get("timestamp", "unknown")

            results.append(
                {
                    "file": os.path.basename(file_path),
                    "task_type": task_type,
                    "dataset_name": dataset_name,
                    "removal": removal,
                    "final_accuracy": final_accuracy,
                    "timestamp": timestamp,
                    "data": data,
                    "file_mtime": os.path.getmtime(file_path),
                }
            )

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return results


def compute_confidence_interval(data, confidence=0.99):
    """Compute confidence interval for given data"""
    n = len(data)
    if n < 2:
        return np.nan, np.nan

    mean = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2.0, n - 1)

    return mean - h, mean + h


def analyze_experiments(results):
    """Analyze experiments and compute statistics"""

    # Group results by task type, dataset name, and removal
    task_groups = {}
    for result in results:
        # Create a composite key for grouping
        group_key = (result["task_type"], result["dataset_name"], result["removal"])

        if group_key not in task_groups:
            task_groups[group_key] = []

        if result["final_accuracy"] is not None:
            task_groups[group_key].append(result["final_accuracy"])

    # Compute statistics for each group
    statistics = []

    for group_key, accuracies in task_groups.items():
        task_type, dataset_name, removal = group_key

        if len(accuracies) == 0:
            continue

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0

        # Compute 99% confidence interval
        ci_lower, ci_upper = compute_confidence_interval(accuracies, confidence=0.99)

        statistics.append(
            {
                "Task Type": task_type,
                "Dataset Name": dataset_name,
                "Removal": removal,
                "Number of Runs": len(accuracies),
                "Mean Accuracy": mean_accuracy,
                "Std Deviation": std_accuracy,
                "99% CI Lower": ci_lower,
                "99% CI Upper": ci_upper,
                "Raw Accuracies": accuracies,
            }
        )

    return statistics


def create_summary_table(statistics):
    """Create formatted summary table"""

    # Create DataFrame
    df = pd.DataFrame(statistics)

    # Remove raw accuracies from display version
    display_df = df.drop("Raw Accuracies", axis=1)

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
        default="../test_results",
        help="Directory containing the test results (default: test_results)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../test_results",
        help="Directory to save the analysis results (default: test_results)",
    )
    args = parser.parse_args()

    print("TA-A-GEM Experiment Analysis")
    print("=" * 50)

    # Load all pickle files
    # args.input_dir = '../test_results/'
    results = load_pickle_files(directory=args.input_dir, num_files=args.num_runs)

    if not results:
        print(f"No pickle files found in {args.input_dir} directory")
        return

    print(
        f"Found {len(results)} result files (analyzing last {args.num_runs} runs only)"
    )

    # Show time range of analyzed files
    if results:
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

    # Show breakdown by task type, dataset name, and removal
    task_counts = {}
    dataset_counts = {}
    removal_counts = {}

    for result in results:
        task_type = result["task_type"]
        dataset_name = result["dataset_name"]
        removal = result["removal"]

        task_counts[task_type] = task_counts.get(task_type, 0) + 1
        dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
        removal_counts[removal] = removal_counts.get(removal, 0) + 1

    print("\nBreakdown by task type:")
    for task_type, count in task_counts.items():
        print(f"  {task_type}: {count} runs")

    print("\nBreakdown by dataset name:")
    for dataset_name, count in dataset_counts.items():
        print(f"  {dataset_name}: {count} runs")

    print("\nBreakdown by removal:")
    for removal, count in removal_counts.items():
        print(f"  {removal}: {count} runs")

    # Analyze experiments
    statistics = analyze_experiments(results)

    if not statistics:
        print("No valid experiment results found")
        return

    # Sort for consistent viewing
    statistics = sorted(statistics, key=lambda a: (a["Task Type"], a["Dataset Name"], a["Removal"]))

    # Create summary table
    display_df, full_df = create_summary_table(statistics)

    # Print results to terminal
    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 100)
    print(display_df.to_string(index=False))

    # Print detailed raw accuracies
    print("\n" + "=" * 100)
    print("DETAILED RESULTS")
    print("=" * 100)
    for stat in statistics:
        print(f"\n{stat['Task Type']} | {stat['Dataset Name']} | {stat['Removal']} ({stat['Number of Runs']} runs):")
        print(f"  Raw accuracies: {[f'{acc:.4f}' for acc in stat['Raw Accuracies']]}")
        print(f"  Mean: {stat['Mean Accuracy']:.4f} ± {stat['Std Deviation']:.4f}")
        print(f"  99% CI: [{stat['99% CI Lower']:.4f}, {stat['99% CI Upper']:.4f}]")

    # Save results to CSV
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
                    "Dataset Name": stat["Dataset Name"],
                    "Removal": stat["Removal"],
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

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()