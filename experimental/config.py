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
DATASET_NAME = 'mnist' # 'mnist' or 'fashion_mnist'
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'
STRATEGY = 3 # 1 for true parallel, 2 for hybrid, 3 for sequential. sequential is best fs
SBATCH = True

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
