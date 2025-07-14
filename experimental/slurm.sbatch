#!/bin/bash
#
# SLURM Batch Script for CPU-bound Parallel Jobs using Job Array
#
# This script reads configuration from config.py to dynamically set paths
# and manage dataset copying to node-local scratch.

# --- SLURM DIRECTIVES (MUST BE AT THE VERY TOP, DO NOT CHANGE PLACEMENT) ---
# These values are parsed by SLURM before the script executes.

#SBATCH --job-name="reuadodd_parallel_taagem"
#SBATCH --output="./sbatch_results/logs/slurm-%A_%a.out" # %A for Job Array ID, %a for Task ID output
#SBATCH --error="./sbatch_results/logs/slurm-%A_%a.err"  # Error log file
#SBATCH --nodes=1             # Nodes per instance. Baseline 1
#SBATCH --ntasks=1            # Each array task runs one main script. Baseline 1
#SBATCH --cpus-per-task=24    # How many CPU cores per instance. Baseline 24
#SBATCH --mem=2G              # Request memory in Gigabytes. Baseline 2G
#SBATCH --time="00:20:00"     # Wall-clock time limit (HH:MM:SS). Baseline 00:20:00
#SBATCH --array=0-19          # Job Array Definition: 20 separate jobs (0 to 19)
#SBATCH --exclude=hpcl4-5,hpcl5-3,hpcl6-1 # Exclude problematic nodes

# If your cluster requires a partition, uncomment and specify it here:
# #SBATCH --partition=hpcl1

# --- END SLURM DIRECTIVES ---


# --- GLOBAL CONFIGURATION (Read from config.py) ---
# These variables are populated by executing Python code to read from config.py.
# This ensures a single source of truth for these dynamic parameters.

# Define Python executable path relative to the script's directory
SCRIPT_DIR=$(dirname "$0")
PYTHON_EXEC="/home/NAS/reuadodd/incremental-learning-reu/venv/bin/python" # Adjust if your venv is located elsewhere

# Add the script's directory to PYTHONPATH so config.py can be imported
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Get parameters from config.py using python -c
DATASET_NAME=$( \
    "$PYTHON_EXEC" -c "import config; print(config.params.get('dataset_name', 'default_dataset'))" \
) || { echo "Error: Failed to get 'dataset_name' from config.py. Check config.py and its structure."; exit 1; }

PROGRAM_OUTPUT_DIR=$( \
    "$PYTHON_EXEC" -c "import config; print(config.params.get('output_base_dir', './sbatch_results'))" \
) || { echo "Error: Failed to get 'output_base_dir' from config.py. Check config.py and its structure."; exit 1; }


# Define the root path for your shared datasets based on your project structure
# This should be the parent directory containing all your dataset folders (e.g., 'mnist', 'cifar10')
SHARED_DATASETS_ROOT="/home/NAS/reuadodd/incremental-learning-reu/datasets"

# Define the base path for node-local scratch (e.g., /tmp, /localscratch, /dev/shm)
# This is where datasets will be copied for faster access.
NODE_LOCAL_SCRATCH_BASE="/home/NAS/reuadodd/tmp"

# --- PROGRAM EXECUTION SETTINGS ---
PYTHON_SCRIPT_PATH="/home/NAS/reuadodd/incremental-learning-reu/experimental/new_main.py" # Absolute path to your main Python script

# --- END GLOBAL CONFIGURATION ---


# --- DYNAMIC DATASET PATH RESOLUTION ---
# Determine the source path of the dataset based on DATASET_NAME from config.py
# This is the path on the shared filesystem.
SOURCE_DATASET_PATH=""
LOCAL_DATASET_SUBDIR="" # The name of the directory once copied to local scratch

case "$DATASET_NAME" in
    "mnist")
        SOURCE_DATASET_PATH="$SHARED_DATASETS_ROOT/mnist"
        ;;
    "fashion_mnist")
        SOURCE_DATASET_PATH="$SHARED_DATASETS_ROOT/fashion_mnist"
        ;;
    "cifar10")
        SOURCE_DATASET_PATH="$SHARED_DATASETS_ROOT/cifar10"
        ;;
    *)
        echo "Error: Unknown dataset_name '$DATASET_NAME' specified in config.py."
        echo "Please add a case for this dataset or correct the name."
        exit 1
        ;;
esac

LOCAL_DATASET_SUBDIR=$(basename "$SOURCE_DATASET_PATH")

# Construct the job-specific node-local scratch directory
# This directory will be unique for each array task and will be cleaned up.
JOB_LOCAL_SCRATCH_DIR="$NODE_LOCAL_SCRATCH_BASE/reuadodd_job_${SLURM_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"


# --- DIRECTORY CREATION ---
# Create necessary directories. These are executable commands.
mkdir -p ./sbatch_results/ # For SLURM's main output/error logs (if not already created by sbatch)
mkdir -p "$PROGRAM_OUTPUT_DIR" # For your program's actual results
mkdir -p "$JOB_LOCAL_SCRATCH_DIR" || { echo "Error: Failed to create local scratch directory $JOB_LOCAL_SCRATCH_DIR. Check path or permissions."; exit 1; }


# --- Copy Dataset to Node-Local Scratch ---
echo "--- Dataset Copy ---"
echo "Configured dataset: $DATASET_NAME"
echo "Source path (shared): $SOURCE_DATASET_PATH"
echo "Destination path (node-local): $JOB_LOCAL_SCRATCH_DIR/$LOCAL_DATASET_SUBDIR"

if [ ! -d "$SOURCE_DATASET_PATH" ]; then
    echo "Error: Source dataset directory '$SOURCE_DATASET_PATH' not found on shared storage."
    exit 1
fi

cp -r "$SOURCE_DATASET_PATH" "$JOB_LOCAL_SCRATCH_DIR/" || \
    { echo "Error: Failed to copy dataset from '$SOURCE_DATASET_PATH' to '$JOB_LOCAL_SCRATCH_DIR/'. Check source path, permissions, or local scratch space."; exit 1; }
echo "Dataset copied successfully."
echo "--------------------"


# --- ENVIRONMENT VARIABLES FOR CACHES/TEMPORARY FILES ---
# These ensure various libraries/tools use your job-specific scratch space
export XDG_CACHE_HOME="$JOB_LOCAL_SCRATCH_DIR/cache"
export TMPDIR="$JOB_LOCAL_SCRATCH_DIR/tmp"
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" || { echo "Error: Failed to create necessary cache/tmp sub-directories in local scratch."; exit 1; }

# Ensure Python runs unbuffered for real-time logging
export PYTHONUNBUFFERED=TRUE


# --- DIAGNOSTIC INFORMATION ---
echo "------------------------------------------------------------"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
echo "Python executable: $PYTHON_EXEC"
echo "Target Python script: $PYTHON_SCRIPT_PATH"
echo "Requested CPU cores (from SLURM): $SLURM_CPUS_PER_TASK"
echo "Dataset being used: $DATASET_NAME (copied to node-local scratch)"
echo "Program outputs will persist in: $PROGRAM_OUTPUT_DIR"
echo "Temporary job-specific scratch for this task: $JOB_LOCAL_SCRATCH_DIR"
echo "------------------------------------------------------------"


# --- BASIC CHECKS ---
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "Error: Python executable '$PYTHON_EXEC' not found. Ensure venv is correctly set up."
    exit 1
fi
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT_PATH' not found."
    exit 1
fi


# --- EXECUTION ---
echo "Executing Python script with arguments:"
# Pass the path to the *copied* dataset on node-local scratch
"$PYTHON_EXEC" "$PYTHON_SCRIPT_PATH" \
    "--slurm-array-task-id" "$SLURM_ARRAY_TASK_ID" \
    "--output-dir" "$PROGRAM_OUTPUT_DIR" \
    "--data-dir" "$JOB_LOCAL_SCRATCH_DIR/$LOCAL_DATASET_SUBDIR" \
    "--run-mode" "slurm"

# Check the exit status of the Python script
PYTHON_EXIT_CODE=$?
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "Error: Python script exited with non-zero status: $PYTHON_EXIT_CODE"
    # You might want to avoid cleaning up if the script failed, for debugging
    # Or add a specific cleanup condition
fi

echo "------------------------------------------------------------"
echo "Task $SLURM_ARRAY_TASK_ID finished at: $(date)"
echo "Final exit status of Python script: $PYTHON_EXIT_CODE"
echo "------------------------------------------------------------"


# --- CLEANUP ---
# Remove the job-specific node-local scratch directory
# This is crucial to avoid filling up /tmp or /localscratch
echo "Removing temporary job-specific local scratch directory: $JOB_LOCAL_SCRATCH_DIR"
rm -rf "$JOB_LOCAL_SCRATCH_DIR"

echo "Job script complete. :)"
