#!/bin/bash

# run_experiments.sh
# Script to run TA-A-GEM experiments 5 times for each task type. Does not parallelize

echo "Starting TA-A-GEM experiments..."
echo "Running 5 iterations for each task type: permutation, rotation, class_split"

# Array of task types
TASK_TYPES=("permutation" "rotation" "class_split")

# Number of runs per task type
NUM_RUNS=5

# Create timestamp for this batch of experiments
BATCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Batch timestamp: $BATCH_TIMESTAMP"

# Counter for total experiments
TOTAL_EXPERIMENTS=$((${#TASK_TYPES[@]} * NUM_RUNS))
CURRENT_EXPERIMENT=0

# Run experiments
for task_type in "${TASK_TYPES[@]}"; do
    echo ""
    echo "=== Running experiments for task type: $task_type ==="

    for run in $(seq 1 $NUM_RUNS); do
        CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
        echo ""
        echo "--- Experiment $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS: $task_type run $run ---"
        echo "Started at: $(date)"

        # Run the experiment
        python main.py --task_type $task_type --no_output

        # Check if the run was successful
        if [ $? -eq 0 ]; then
            echo "Completed: $task_type run $run at $(date)"
        else
            echo "ERROR: Failed $task_type run $run at $(date)"
        fi
    done
done

echo ""
echo "=== All experiments completed ==="
echo "Finished at: $(date)"
echo ""
echo "Running analysis script..."

# Run the analysis script
python visualization_analysis/retro_stats.py

echo "Analysis complete!"
echo "Results saved to test_results/ directory"
