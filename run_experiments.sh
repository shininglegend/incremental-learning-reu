#!/bin/bash

# run_experiments.sh
# Script to run TA-A-GEM experiments 5 times for each task type using SLURM

echo "Starting TA-A-GEM experiments with SLURM..."
echo "Running 5 iterations for each task type: permutation, rotation, class_split"

# Array of task types
TASK_TYPES=("permutation" "rotation" "class_split")

# Number of runs per task type
NUM_RUNS=5

# Create timestamp for this batch of experiments
BATCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Batch timestamp: $BATCH_TIMESTAMP"

# Create directory for SLURM job scripts and logs
mkdir -p slurm_jobs
mkdir -p slurm_logs

# Array to store job IDs
JOB_IDS=()

# Counter for total experiments
TOTAL_EXPERIMENTS=$((${#TASK_TYPES[@]} * NUM_RUNS))
echo "Total experiments to run: $TOTAL_EXPERIMENTS"

# Submit all jobs to SLURM
for task_type in "${TASK_TYPES[@]}"; do
    echo ""
    echo "=== Submitting jobs for task type: $task_type ==="

    for run in $(seq 1 $NUM_RUNS); do
        JOB_NAME="taagem_${task_type}_run${run}_${BATCH_TIMESTAMP}"

        # Create SLURM job script
        cat > slurm_jobs/${JOB_NAME}.sh << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=slurm_logs/${JOB_NAME}.out
#SBATCH --error=slurm_logs/${JOB_NAME}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=4G

echo "Starting experiment: $task_type run $run"
echo "Started at: \$(date)"
echo "Running on node: \$(hostname)"

# Run the experiment
python main.py --task_type $task_type

# Check if the run was successful
if [ \$? -eq 0 ]; then
    echo "Completed: $task_type run $run at \$(date)"
else
    echo "ERROR: Failed $task_type run $run at \$(date)"
    exit 1
fi
EOF

        # Submit the job
        JOB_ID=$(sbatch slurm_jobs/${JOB_NAME}.sh | awk '{print $4}')
        JOB_IDS+=($JOB_ID)
        echo "Submitted job $JOB_ID: $task_type run $run"
    done
done

echo ""
echo "All jobs submitted. Job IDs: ${JOB_IDS[@]}"
echo "Waiting for all jobs to complete..."

# Wait for all jobs to complete
for job_id in "${JOB_IDS[@]}"; do
    echo "Waiting for job $job_id..."
    while squeue -j $job_id 2>/dev/null | grep -q $job_id; do
        sleep 10
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
echo "SLURM logs saved to slurm_logs/ directory"
