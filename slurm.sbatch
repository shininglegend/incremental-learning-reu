#!/bin/bash
#SBATCH --job-name="parallel_taagem"
#SBATCH --output="./sbatch_results/slurm-%A_%a.out" # %A for Job Array ID, %a for Task ID output
#SBATCH --error="./sbatch_results/slurm-%A_%a.err"  # Error log file
#SBATCH --nodes=1 # Nodes per instance. 1 node = 24 CPUs. Baseline 1
#SBATCH --ntasks=1 # Each array task runs one main script. Baseline 1
#SBATCH --cpus-per-task=24 # How many CPU cores per instance. Baseline 24
#SBATCH --mem=2G  # Request memory in Gigabytes. One new_main.py job takes about 1.2G RAM. Request a little extra.
#SBATCH --time="00:30:00" # 00:12:41 baseline. Request a little extra.
#SBATCH --array=0-4 # Adjust the range based on how many runs you want. 0-4 for 5 runs, 0-19 for 20 runs
#SBATCH --exclude=hpcl4-5,hpcl5-3,hpcl6-1,hpcl4-1,hpcl4-2,hpcl4-3 # these guys are bad
# #SBATCH --partition="$PARTITION_NAME"


# A Slurm batch script for running the program many times in parallel across CPUs

# ------- GLOBAL CONFIG (CUSTOMIZE WITH YOUR FILE LOCATIONS) -------
HOME_DIR="/home/NAS/$USER/incremental-learning-reu"
PYTHON_SCRIPT="$HOME_DIR/main.py" # Or wherever you put main.py

# Define Python executable path relative to the script's directory
PYTHON_EXEC=$3
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "Error: Python executable '$PYTHON_EXEC' not found."
    exit 1
fi

# General Job Settings
JOB_NAME="parallel_taagem"          # A descriptive name for your job
TIME_LIMIT="00:20:00"            # Max time for each task (HH:MM:SS)
TASK_TYPE=$1 # Type of task: "permutation", "class_split", "rotation"
DATASET_NAME=$2 # mnist or fashion_mnist
if [[ -z "$TASK_TYPE" || -z "$DATASET_NAME" ]]; then
    echo "Usage: $0 <task_type> <dataset_name>"
    echo "Example: $0 permutation mnist"
    exit 1
fi


# Resource Requests per Program Instance (i.e., per Array Task)
NODES_PER_INSTANCE=1             # Number of nodes each program instance needs. baseline 1
CORES_PER_INSTANCE=24            # Number of CPU cores each program instance needs. 24 for a full node
MEM_PER_INSTANCE_GB=2            # Memory in GB each program instance needs. 2G baseline

# --- EXECUTION ---

#JOB-SPECIFIC Directory for all sbatch stuff
JOB_SCRATCH_DIR="$HOME_DIR/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "DEBUG: SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "DEBUG: Attempting to create directory: \"$JOB_SCRATCH_DIR\""

mkdir -p "$JOB_SCRATCH_DIR" || { echo "Error: Failed to create scratch directory $JOB_SCRATCH_DIR. Check path or permissions."; exit 1; }

SOURCE_DATASET_PATH="$HOME_DIR/datasets"

echo "Copying dataset from $SOURCE_DATASET_PATH to $JOB_SCRATCH_DIR/mnist-dataset/"
cp -r "$SOURCE_DATASET_PATH" "$JOB_SCRATCH_DIR/mnist-dataset/"
if [ $? -ne 0 ]; then
    echo "Error: Failed to copy dataset. Check SOURCE_DATASET_PATH and permissions."
    exit 1
fi
echo "Dataset copied successfully."


# Kaggle is being mean :(
export KAGGLE_HOME="$JOB_SCRATCH_DIR/kaggle_data"
export KAGGLE_CONFIG_DIR="$JOB_SCRATCH_DIR/.kaggle"
export KAGGLE_DATASET_DIR="$JOB_SCRATCH_DIR/datasets"

export XDG_CACHE_HOME="$JOB_SCRATCH_DIR/cache"
export TMPDIR="$JOB_SCRATCH_DIR/tmp"
mkdir -p "$KAGGLE_CONFIG_DIR" "$XDG_CACHE_HOME" "$TMPDIR" || { echo "Error: Failed to create necessary sub-directories for caches."; exit 1; }


# Define Python executable path relative to the script's directory
SCRIPT_DIR=$(dirname "$0")

# Ensure stdout/stderr are unbuffered for real-time logging
export PYTHONUNBUFFERED=TRUE

# Basic checks
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found."
    exit 1
fi

echo "------------------------------------------------------------"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
echo "Python executable: $PYTHON_EXEC"
echo "Target Python script: $PYTHON_SCRIPT"
echo "This task requested $CORES_PER_INSTANCE CPU cores."
echo "------------------------------------------------------------"


# --- Debugging variables ---
echo "--- SLURM Variable Debugging ---"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID" # This is the ID of the parent array job
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "TOTAL_RUNS (from sbatch): $TOTAL_RUNS"
echo "--- End SLURM Variable Debugging ---"

# Execute your Python script
echo "Executing Python script with arguments:"
echo "$PYTHON_EXEC" "$PYTHON_SCRIPT" --output_dir "$JOB_SCRATCH_DIR" --data_dir "$JOB_SCRATCH_DIR/mnist-dataset" --task_type "$TASK_TYPE" --dataset "$DATASET_NAME" --no_verbose

# Execute your Python script directly for this array task.
# You can pass the array task ID as an argument to your Python script if it needs it.
"$PYTHON_EXEC" "$PYTHON_SCRIPT" --output_dir "$JOB_SCRATCH_DIR" --data_dir "$JOB_SCRATCH_DIR/mnist-dataset" --task_type "$TASK_TYPE" --dataset "$DATASET_NAME" --no_verbose

echo "------------------------------------------------------------"
echo "Task $SLURM_ARRAY_TASK_ID finished at: $(date)"
echo "------------------------------------------------------------"

# --- CLEANUP ---
echo "Copying output files to test_results directory"
mkdir -p "$HOME_DIR/test_results/$4"
for file in "$JOB_SCRATCH_DIR"/*; do
    [ -f "$file" ] || continue
    basename=$(basename "$file")
    dest="$HOME_DIR/test_results/$4/$basename"
    counter=1
    while [ -e "$dest" ]; do
        name="${basename%.*}"
        ext="${basename##*.}"
        [ "$name" = "$ext" ] && dest="$HOME_DIR/test_results/$4/${basename}_$counter" || dest="$HOME_DIR/test_results/$4/${name}_$counter.$ext"
        ((counter++))
    done
    cp "$file" "$dest"
done
rm -rf "$JOB_SCRATCH_DIR"

echo "Job script complete. :)"
