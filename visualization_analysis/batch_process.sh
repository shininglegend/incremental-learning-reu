#!/bin/bash

# Batch process script for retro_stats.py
# Usage: ./batch_process.sh <input_dir> <output_dir>

# PYTHON_EXEC="./venv/bin/python" # Uncomment for pip
PYTHON_EXEC="./env/bin/python" # Uncomment for conda

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir>"
    echo "  input_dir: Directory containing subfolders to process"
    echo "  output_dir: Directory to save all detailed results"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Counter for processed folders
processed_count=0
skipped_folders=()

echo "Batch processing folders in: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Processing folders that start with '4'..."
echo

# Process each subdirectory that starts with 4
for folder in "$INPUT_DIR"/4*; do
    # Check if it's actually a directory
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")
        echo "Processing folder: $folder_name"

        # Check if folder contains any .pkl files
        pkl_count=$(find "$folder" -name "*.pkl" | wc -l)
        if [ "$pkl_count" -eq 0 ]; then
            echo "  Warning: No .pkl files found in $folder_name, skipping"
            skipped_folders+=("$folder_name (no pkl files)")
            continue
        fi

        # Assert exactly 15 pickle files
        if [ "$pkl_count" -ne 15 ]; then
            echo "  Error: Expected 15 pickle files, found $pkl_count in $folder_name, skipping"
            skipped_folders+=("$folder_name ($pkl_count pkl files)")
            continue
        fi

        echo "  Found $pkl_count pickle files"

        # Run retro_stats.py on this folder
        $PYTHON_EXEC "$SCRIPT_DIR/retro_stats.py" \
            --input_dir "$folder" \
            --output_dir "$OUTPUT_DIR" \
            --num_runs 50 \
            --no_verbose

        if [ $? -eq 0 ]; then
            echo "  ✓ Successfully processed $folder_name"
            ((processed_count++))
        else
            echo "  ✗ Error processing $folder_name"
            exit 1
        fi
        echo
    else
        echo "Skipping non-directory: $(basename "$folder")"
        skipped_folders+=("$(basename "$folder") (not a directory)")
    fi
done

echo "Batch processing complete!"
echo "Processed: $processed_count folders"
if [ ${#skipped_folders[@]} -gt 0 ]; then
    echo "Skipped folders:"
    for folder in "${skipped_folders[@]}"; do
        echo "  - $folder"
    done
else
    echo "No folders skipped"
fi
echo "Results saved to: $OUTPUT_DIR"

# List the generated files
# echo
# echo "Generated files:"
# ls -la "$OUTPUT_DIR"/detailed_results-*.csv 2>/dev/null || echo "No detailed results files found"
