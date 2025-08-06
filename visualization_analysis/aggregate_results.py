#!./env/bin/python
"""
Aggregator script for incremental learning test results.
Processes detailed CSVs and generates summary spreadsheets grouped by dataset and task introduction type.
"""

import os
import pandas as pd
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


def process_csv_file(csv_path: Path) -> Tuple[str, pd.DataFrame]:
    """Process a single CSV file and return folder name and data."""
    # Extract folder name from filename
    filename = Path(csv_path).stem
    # Remove 'detailed_results-' prefix if present
    folder_name = filename.replace("detailed_results-", "")
    df = pd.read_csv(csv_path)
    return folder_name, df


def aggregate_results(input_dir: str, output_dir: str):
    """Main aggregation function."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_files = list(input_path.glob("detailed_results-*.csv"))

    if not csv_files:
        print(f"No detailed CSV files found in {input_dir}")
        return

    results_by_dataset_task = {}

    for csv_file in csv_files:
        folder_name, df = process_csv_file(csv_file)

        if df.empty:
            continue

        dataset = extract_dataset_from_folder(folder_name)

        for task_intro in df["task_introduction"].unique():
            if dataset not in results_by_dataset_task:
                results_by_dataset_task[dataset] = {}
            if task_intro not in results_by_dataset_task[dataset]:
                results_by_dataset_task[dataset][task_intro] = []

            task_df = df[df["task_introduction"] == task_intro]

            folder_results = {}
            for task_type in task_df["task_type"].unique():
                type_df = task_df[task_df["task_type"] == task_type]
                accuracies = type_df["accuracy"].tolist()
                forgettings = type_df["forgetting"].tolist()
                first_tasks = type_df["first_task_accuracy"].tolist()

                folder_results[task_type] = {
                    "accuracies": accuracies,
                    "forgettings": forgettings,
                    "first_tasks": first_tasks,
                }

            results_by_dataset_task[dataset][task_intro].append(
                {"folder_name": folder_name, "results": folder_results}
            )

    for dataset, task_intros in results_by_dataset_task.items():
        for task_intro, folder_data in task_intros.items():
            generate_summary_spreadsheet(dataset, task_intro, folder_data, output_path)


def generate_summary_spreadsheet(
    dataset: str, task_intro: str, folder_data: List[Dict], output_path: Path
):
    """Generate a summary spreadsheet for a specific dataset and task introduction type."""
    summary_rows = []

    for folder_info in folder_data:
        folder_name = folder_info["folder_name"]
        results = folder_info["results"]

        for task_type, task_data in results.items():
            accuracies = task_data["accuracies"]
            forgettings = task_data["forgettings"]
            first_tasks = task_data["first_tasks"]

            row = {
                "Folder": folder_name,
                "Dataset": dataset,
                "Task": extract_task_type_mapping(task_type),
            }

            # Add accuracy results with 6 decimal places
            for i in range(5):
                if i < len(accuracies):
                    row[f"Accuracy_{i+1}"] = f"{accuracies[i]:.6f}"
                else:
                    row[f"Accuracy_{i+1}"] = ""

            # Add forgetting results with 6 decimal places
            for i in range(5):
                if i < len(forgettings):
                    row[f"Forgetting_{i+1}"] = f"{forgettings[i]:.6f}"
                else:
                    row[f"Forgetting_{i+1}"] = ""

            # Add first task results with 6 decimal places
            for i in range(5):
                if i < len(first_tasks):
                    row[f"FirstTask_{i+1}"] = f"{first_tasks[i]:.6f}"
                else:
                    row[f"FirstTask_{i+1}"] = ""

            summary_rows.append(row)

    if not summary_rows:
        return

    df = pd.DataFrame(summary_rows)
    df = df.sort_values(["Folder", "Task"])

    filename = f"{dataset.lower().replace('-', '_')}_{task_intro}_summary.csv"
    output_file = output_path / filename

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
