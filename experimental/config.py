import argparse
import os

QUICK_TEST_MODE = False
NUM_CLASSES = 10
INPUT_DIM = 784
HIDDEN_DIM = 200
MEMORY_SIZE_Q = 10
MEMORY_SIZE_P = 3
BATCH_SIZE = 50 if QUICK_TEST_MODE else 10
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
NUM_TASKS = 2 if QUICK_TEST_MODE else 5
TASK_TYPE = 'permutation' # 'permutation', 'rotation', or 'class split'
NUM_POOLS = 10
CLUSTERS_PER_POOL = 10
DATASET_NAME = 'fashion_mnist' # 'mnist' or 'fashion_mnist'
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'
STRATEGY = 3 # 1 for true parallel, 2 for hybrid, 3 for sequential. sequential is best fs
SBATCH = True # Set to False when running files directly, True when running w/ SBATCH


params = {
    'input_dim': INPUT_DIM,
    'hidden_dim': HIDDEN_DIM,
    'num_classes': NUM_CLASSES,
    'memory_size_q': CLUSTERS_PER_POOL,
    'memory_size_p': MEMORY_SIZE_P,
    'num_pools': NUM_POOLS,
    'task_type': TASK_TYPE,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'num_epochs': NUM_EPOCHS,
    'num_tasks': NUM_TASKS,
    'quick_test_mode': QUICK_TEST_MODE,
    'device': DEVICE,
    'dataset_name': DATASET_NAME,
    'strategy': STRATEGY,
    'sbatch': SBATCH,
    }


def get_params():
    return params


# Parser for sbatch
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a parallel task.")

    # Check if a 'params' dictionary is needed (if it's defined elsewhere)
    # For now, let's assume if it's run via sbatch, it will have these args
    # You might pass `params` into this function if its source is external

    # These args are expected when running via sbatch
    parser.add_argument("--task-id", type=int, default=None,  # default to None for local runs
                        help="SLURM Array Task ID")
    parser.add_argument("--output-dir", type=str, default="output_results",
                        help="Directory to save output results.")
    parser.add_argument("--data-dir", type=str, default="local_data",
                        help="Directory where datasets are located.")
    # Add a flag to explicitly indicate run mode (useful if task-id is optional)
    parser.add_argument("--run-mode", type=str, default="local",
                        help="Mode of execution ('slurm' or 'local').")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Data will be loaded from: {args.data_dir}")  # Debug print
    print(f"Output will be saved to: {args.output_dir}")  # Debug print
    print(f"Task ID (if applicable): {args.task_id}")

    return args  # Return the parsed arguments
