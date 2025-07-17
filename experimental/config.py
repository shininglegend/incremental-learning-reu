import argparse
import os

'''   You can touch these   '''
QUICK_TEST_MODE = False
TASK_TYPE = 'permutation' # 'permutation', 'rotation', or 'class_split'
DATASET_NAME = 'mnist' # 'mnist' or 'fashion_mnist'
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'
SBATCH = True # Set to False when running files directly, True when running w/ SBATCH
if SBATCH:
    OUTPUT_DIR = "./what_the"
else:
    OUTPUT_DIR = "./output_results"

# To add a new removal, make sure you update clustering_mechs!
REMOVAL = 'remove_based_on_mean' # 'remove_oldest' or 'remove_based_on_mean'.
# added 'remove_random' and 'remove_furthest_from_mean' as experimental options

'''   Experimental stuff   '''
RECLUSTERING_FREQ = 3 # how many times you want to recluster


"""   Don't touch these, probably   """
NUM_CLASSES = 10
INPUT_DIM = 784
HIDDEN_DIM = 200
MEMORY_SIZE_Q = 10
MEMORY_SIZE_P = 3
BATCH_SIZE = 50 if QUICK_TEST_MODE else 10
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
NUM_TASKS = 2 if QUICK_TEST_MODE else 5
NUM_POOLS = 10
CLUSTERS_PER_POOL = 10



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
    'sbatch': SBATCH,
    'output_dir': OUTPUT_DIR,
    'removal': REMOVAL,
    'reclustering_freq': RECLUSTERING_FREQ,
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
    parser.add_argument("--output-dir", type=str, default="output_results",
                        help="Directory to save output results.")
    parser.add_argument("--data-dir", type=str, default="local_data",
                         help="Directory where datasets are located.")
    parser.add_argument("--run-mode", type=str, default="local",
                        help="Mode of execution ('slurm' or 'local').")
    parser.add_argument("--slurm-array-task-id", type=int, default=None,
                        help="SLURM Array Task ID for this job array element.")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    # os.makedirs(args.output_dir, exist_ok=True)

    return args  # Return the parsed arguments
