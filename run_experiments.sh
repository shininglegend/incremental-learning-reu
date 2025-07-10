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

echo "=========================================="
echo "JOB START: $task_type run $run"
echo "Started at: \$(date)"
echo "Running on node: \$(hostname)"
echo "Job ID: \$SLURM_JOB_ID"
echo "=========================================="

# Run the experiment
python main.py --task_type $task_type

# Check if the run was successful
if [ \$? -eq 0 ]; then
    echo "=========================================="
    echo "JOB COMPLETED: $task_type run $run"
    echo "Completed at: \$(date)"
    echo "Job ID: \$SLURM_JOB_ID"
    echo "=========================================="
else
    echo "=========================================="
    echo "JOB FAILED: $task_type run $run"
    echo "Failed at: \$(date)"
    echo "Job ID: \$SLURM_JOB_ID"
    echo "=========================================="
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

# Function to get job status
get_job_status() {
    local job_id=$1
    squeue -j $job_id -h -o "%T" 2>/dev/null || echo "COMPLETED"
}

# Function to display current job status
display_job_status() {
    echo ""
    echo "=== Job Status Update at $(date) ==="
    local running_jobs=0
    local pending_jobs=0
    local completed_jobs=0

    for job_id in "${JOB_IDS[@]}"; do
        status=$(get_job_status $job_id)
        case $status in
            "RUNNING")
                running_jobs=$((running_jobs + 1))
                # Get job info
                job_info=$(squeue -j $job_id -h -o "%j %N" 2>/dev/null)
                echo "  RUNNING: Job $job_id ($job_info)"
                ;;
            "PENDING")
                pending_jobs=$((pending_jobs + 1))
                job_info=$(squeue -j $job_id -h -o "%j" 2>/dev/null)
                echo "  PENDING: Job $job_id ($job_info)"
                ;;
            "COMPLETED"|"")
                completed_jobs=$((completed_jobs + 1))
                ;;
        esac
    done

    echo "Summary: $running_jobs running, $pending_jobs pending, $completed_jobs completed"
    echo "=========================================="
}

# Wait for all jobs to complete with periodic updates
while true; do
    all_completed=true
    for job_id in "${JOB_IDS[@]}"; do
        if squeue -j $job_id 2>/dev/null | grep -q $job_id; then
            all_completed=false
            break
        fi
    done

    if $all_completed; then
        break
    fi

    display_job_status
    sleep 30
done

echo ""
echo "=== All jobs completed at $(date) ==="

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
