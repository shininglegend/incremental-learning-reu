#!./env/bin/python
"""
Aggregator script for incremental learning test results.
Processes detailed CSVs and generates summary spreadsheets grouped by dataset and task introduction type.
"""

import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def extract_dataset_from_folder(folder_name: str) -> str:
    """Extract dataset name from folder name."""
    # Look for common dataset patterns
    if "mnist" in folder_name.lower():
        return "MNIST"
    elif "fashion" in folder_name.lower():
        return "Fashion-MNIST"
    elif "cifar" in folder_name.lower():
        return "CIFAR-10"
    else:
        # Default fallback
        return "Unknown"


def extract_task_type_mapping(task_type: str) -> str:
    """Map task type to standard format."""
    mapping = {
        "class_split": "Class Split",
        "permutation": "Permutation",
        "rotation": "Rotation",
    }
    return mapping.get(task_type, task_type.title())


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute mean, std dev, min, max for a list of values."""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}

    arr = np.array(values)
    return {
        "mean": np.mean(arr),
        "std": np.std(arr, ddof=1) if len(arr) > 1 else 0,
        "min": np.min(arr),
        "max": np.max(arr),
    }


def process_csv_file(csv_path: str) -> Tuple[str, pd.DataFrame]:
    """Process a single CSV file and return folder name and data."""
    # Extract folder name from filename
    filename = Path(csv_path).stem
    # Remove 'detailed_results-' prefix if present
    folder_name = filename.replace("detailed_results-", "")

    # Read CSV
    df = pd.read_csv(csv_path)

    return folder_name, df


def aggregate_results(input_dir: str, output_dir: str):
    """Main aggregation function."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all detailed CSV files
    csv_files = list(input_path.glob("detailed_results-*.csv"))

    if not csv_files:
        print(f"No detailed CSV files found in {input_dir}")
        return

    # Process all CSV files and organize by dataset and task_introduction
    results_by_dataset_task = (
        {}
    )  # dataset -> task_introduction -> list of folder results

    for csv_file in csv_files:
        folder_name, df = process_csv_file(csv_file)

        if df.empty:
            continue

        # Extract dataset from folder name
        dataset = extract_dataset_from_folder(folder_name)

        # Group by task_introduction and task_type within this folder
        for task_intro in df["task_introduction"].unique():
            if dataset not in results_by_dataset_task:
                results_by_dataset_task[dataset] = {}
            if task_intro not in results_by_dataset_task[dataset]:
                results_by_dataset_task[dataset][task_intro] = []

            # Get data for this task_introduction
            task_df = df[df["task_introduction"] == task_intro]

            # Group by task_type and compute statistics
            folder_results = {}
            for task_type in task_df["task_type"].unique():
                type_df = task_df[task_df["task_type"] == task_type]
                accuracies = type_df["accuracy"].tolist()

                folder_results[task_type] = {
                    "accuracies": accuracies,
                    "stats": compute_statistics(accuracies),
                }

            results_by_dataset_task[dataset][task_intro].append(
                {"folder_name": folder_name, "results": folder_results}
            )

    # Generate summary spreadsheets
    for dataset, task_intros in results_by_dataset_task.items():
        for task_intro, folder_data in task_intros.items():
            generate_summary_spreadsheet(dataset, task_intro, folder_data, output_path)


def generate_summary_spreadsheet(
    dataset: str, task_intro: str, folder_data: List[Dict], output_path: Path
):
    """Generate a summary spreadsheet for a specific dataset and task introduction type."""

    # Prepare data for the spreadsheet
    summary_rows = []

    for folder_info in folder_data:
        folder_name = folder_info["folder_name"]
        results = folder_info["results"]

        for task_type, task_data in results.items():
            stats = task_data["stats"]
            accuracies = task_data["accuracies"]

            # Create row similar to the example format
            row = {
                "Folder": folder_name,
                "Dataset": dataset,
                "Task": extract_task_type_mapping(task_type),
                "Mean": stats["mean"],
                "StdDev": stats["std"],
                "Range_Min": stats["min"],  # Will be filtered out later
                "Range_Max": stats["max"],  # Will be filtered out later
            }

            # Add individual run results (up to 5)
            for i in range(5):
                if i < len(accuracies):
                    row[f"{i+1}"] = accuracies[i]
                else:
                    row[f"{i+1}"] = ""

            # Add raw values as comma-separated string
            # row['Raw values'] = ','.join([f'{acc:.4f}' for acc in accuracies])

            summary_rows.append(row)

    if not summary_rows:
        return

    # Create DataFrame
    df = pd.DataFrame(summary_rows)

    # Remove the columns we don't want
    columns_to_remove = ["Range_Min", "Range_Max"]
    df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])

    # Sort by Method (folder name) and Task
    df = df.sort_values(["Folder", "Task"])

    # Generate filename
    filename = f"{dataset.lower().replace('-', '_')}_{task_intro}_summary.csv"
    output_file = output_path / filename

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Generated: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aggregate incremental learning test results"
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory containing detailed CSV files"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save summary spreadsheets"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return

    aggregate_results(args.input_dir, args.output_dir)
    print("Aggregation complete!")


if __name__ == "__main__":
    main()
