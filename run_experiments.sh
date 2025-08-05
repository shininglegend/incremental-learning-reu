#!/bin/bash

# run_experiments.sh

# Array of task types
TASK_TYPES=("permutation" "rotation" "class_split")

# Number of runs per task type
NUM_RUNS=5

# Create timestamp for this batch of experiments
BATCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Batch timestamp: $BATCH_TIMESTAMP"

# Choose dataset based on user input (restrict to mnist, fashion_mnist, and cifar10)
if [[ "$1" != "mnist" && "$1" != "fashion_mnist" && "$1" != "cifar10" ]]; then
    echo "Usage: $0 <dataset_name> <save_location>"
    echo "Example: $0 mnist test-new-remove-one"
    exit 1
fi
DATASET_NAME=$1

# Add a folder to save test results based on user input
if [[ -z "$2" ]]; then
    echo "Usage: $0 <dataset_name> <save_location>"
    echo "Example: $0 mnist test-new-remove-one"
    exit 1
fi
if [[ ! -d "test_results/$2" ]]; then
    mkdir -p "test_results/$2"
fi
SAVE_LOCATION="test_results/$2"
echo "Results will be saved to: $SAVE_LOCATION"

echo "Starting experiments..."
echo "Running 5 iterations for each task type: ${TASK_TYPES[*]}"


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
        python main.py --task_type $task_type --output_dir "$SAVE_LOCATION" --dataset "$DATASET_NAME" --experiment_name "$2"

        # Check if the run was successful
        if [ $? -eq 0 ]; then
            echo "Completed: $task_type run $run at $(date)"
        else
            echo "ERROR: Failed $task_type run $run at $(date)"
            exit 1
        fi
    done
done

echo ""
echo "=== All experiments completed ==="
echo "Finished at: $(date)"
echo ""
echo "Running analysis script..."

# Run the analysis script
python visualization_analysis/retro_stats.py --input_dir "$SAVE_LOCATION" --output_dir "$SAVE_LOCATION"

echo "Analysis complete!"
echo "Results saved to $SAVE_LOCATION"
