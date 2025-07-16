#!/bin/bash

# run_experiments_batch.sh
# Script to run TA-A-GEM experiments in parallel using SBATCH

# Define Python executable path relative to the script's directory
# If you have a virtual environment, adjust the path accordingly.
# You can tell conda to put the env in the same directory as the script by using:
# conda create --prefix ./env -f linux_env.yml

#PYTHON_EXEC="incremental-learning-reu/venv/bin/python" # Uncomment for pip venv
PYTHON_EXEC="env/bin/python" # Uncomment for conda env

# Check if the Python executable exists
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "Error: Python executable '$PYTHON_EXEC' not found."
    exit 1
fi

dataset_name=$1
if [[ -z "$dataset_name" ]]; then
    echo "Usage: $0 <dataset_name>"
    echo "Example: $0 mnist"
    exit 1
fi

echo "Starting TA-A-GEM parallel experiments..."
echo "Running 5 iterations for each task type: permutation, rotation, class_split"
echo "Dataset: $dataset_name"

# Array of task types
TASK_TYPES=("permutation" "rotation" "class_split")

# Create timestamp for this batch of experiments
BATCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Batch timestamp: $BATCH_TIMESTAMP"

# Submit jobs for each task type
for task_type in "${TASK_TYPES[@]}"; do
    echo "Submitting job for task type: $task_type with dataset: $dataset_name"

    # Submit job
    JOB_OUTPUT=$(sbatch slurm.sbatch $task_type $dataset_name "$PYTHON_EXEC")
    JOB_ID=$(echo $JOB_OUTPUT | grep -o '[0-9]*')

    echo "Job ID $JOB_ID submitted for $task_type"
done

echo ""
echo "All jobs submitted."
echo "Monitoring job progress..."

# Monitor jobs until completion
while true; do
    # Check if user has any running jobs
    RUNNING_JOBS=$(squeue -u $USER --noheader 2>/dev/null | wc -l)

    if [[ $RUNNING_JOBS -eq 0 ]]; then
        echo "All jobs completed!"
        break
    fi

    echo "Jobs still running: $RUNNING_JOBS ($(date))"
    sleep 30
done

echo ""
echo "=== All experiments completed ==="
echo "Finished at: $(date)"
echo ""
echo "Running analysis script..."

# Run the analysis script
"$PYTHON_EXEC" visualization_analysis/retro_stats.py
echo "Results saved to test_results/ directory"
