# TA-A-GEM: Task Agnostic Averaged Gradient Episodic Memory

This repository contains an implementation of **Task-Agnostic Averaged Gradient Episodic Memory (TA-A-GEM)**, a continual learning approach that combines A-GEM's gradient projection with clustering for episodic memory management. It is based on the paper by Lamers et al. (2023).

## Overview

TA-A-GEM addresses catastrophic forgetting in continual learning by:

- Using **gradient projection** (A-GEM) to prevent interference between tasks
- Implementing **task agnostic clustering** to maintain representative samples in episodic memory
- Supporting **multi-pool architecture** for different task types
- Providing comprehensive **visualization and analysis** tools

### Configuration:

- **`utils/default.yaml`**: Change settings here as you wish.

## Architecture

The implementation consists of several key components:

### Core Components

- **`main.py`**: Main training loop and experiment orchestration
- **`agem.py`**: A-GEM gradient projection implementation
- **`clustering.py`**: Multi-pool clustering memory management
  - **Multi-pool management**: Separate clustering pools per class/task type
- **`load_dataset.py`**: Abstract data loading and task generation
- **`visualization_analysis.py`**: Analysis, visualization, and timing tools
- **`clustering_gemini.py`** for core clustering algorithms

### Dependencies

- See the \*\_env.yml files for dependencies.

## Installation

### Environment Setup

Choose the appropriate environment file based on your system:

**For Linux/CUDA systems:**

```bash
conda env create -f linux_env.yml
conda activate ta-env
```

**For macOS systems:**

```bash
conda env create -f mac_env.yml
conda activate ta-a-gem
```

**For windows systems:**

We have not tested on windows. Try the mac env above, or create your own and install the needed packages. Feel free to open a PR if you have one that works!

### Dataset Setup

The code automatically downloads the datasets via kagglehub. If you encounter issues:

1. Manually download the correct dataset from: https://www.kaggle.com/
2. Update the `path` variable in the relevant dataset file with your path.

## Usage

### Basic Execution

**Quick Test Mode (Recommended for initial testing):**

```bash
python3 main.py --lite
```

- Runs with reduced dataset
- Detailed loss logging per batch
- Faster execution for debugging

**Full Experiment Mode:**

```bash
python3 main.py
```

- Runs 5 tasks with full dataset
- Progress bar display
- Complete experimental setup

### Configuration Parameters

Key parameters can be modified in `config/default.yaml` file or passed in via command line args (do `python3 main.py -h` for options.)

#### Task Configuration

```yaml
NUM_TASKS 2 if QUICK_TEST_MODE else 5  # Number of continual learning tasks
TASK_TYPE = 'permutation'  # 'permutation', 'rotation', or 'class_split'
```

#### Model Architecture

```python
INPUT_DIM = 784      # flattened (28*28)
NUM_CLASSES = 10     # incoming classes
HIDDEN_DIM = 200     # Hidden layer size for MLP (as per paper)
```

#### Memory Configuration

```python
MEMORY_SIZE_Q = 10   # Clusters per pool (Q)
MEMORY_SIZE_P = 3    # Max samples per cluster (P)
NUM_POOLS = 10 or 5       # Number of memory pools (tied to task_type)
```

#### Training Parameters

```python
BATCH_SIZE = 10      # Batch size (as per paper)
LEARNING_RATE = 1e-3 # Learning rate
USE_LEARNING_RATE_SCHEDULER = True  # Whether to use the learning rate scheduler described in the paper
```

### Task Types

#### 1. Permutation Tasks

```python
TASK_TYPE = 'permutation'
NUM_POOLS = 10  # 10 pools for permutation tasks
```

- Applies random pixel permutations to dataset images
- Each task uses a different fixed permutation
- Tests model's ability to handle input transformation

#### 2. Rotation Tasks

```python
TASK_TYPE = 'rotation'
NUM_POOLS = 10  # 10 pools for rotation tasks
```

- Rotates dataset images by different angles
- Angles distributed evenly across tasks
- Tests spatial invariance learning

#### 3. Class Split Tasks

```python
TASK_TYPE = 'class_split'
NUM_POOLS = 2   # 2 pools for class split tasks
```

- Splits 10 dataset classes across tasks
- Each task contains subset of classes
- Tests class-incremental learning

## Implementation Details

### A-GEM Gradient Projection

The `AGEMHandler` class implements the core A-GEM algorithm:

```python
def optimize(self, data, labels, memory_samples=None):
    # 1. Compute gradient on current task
    # 2. If memory exists, compute reference gradient
    # 3. Project current gradient to not increase memory loss
    # 4. Apply projected gradient
```

**Key features:**

- Gradient computation without model state corruption (hopefully)
- Dot product-based projection

### Multi-Pool Clustering Memory

The `ClusteringMemory` class manages Task Agnostic episodic memory:

```python
def add_sample(self, sample_data, sample_label):
    # Routes samples to appropriate pool based on label
    pool = self._get_or_create_pool(sample_label)
    pool.add(sample_data, sample_label)
```

**Architecture:**

- Separate clustering pools per class/task
- Dynamic pool creation as new classes encountered, limit NUM_POOLS
- Configurable cluster count (Q) and samples per cluster (P)

### Task Data Generation

The `prepare_domain_incremental_data` function creates task-specific datasets:

**Permutation Tasks:**

- Ensures each task gets different images
- Generates unique random permutations
- Applies to flattened dataset images

**Rotation Tasks:**

- Ensures each task gets different images
- Calculates evenly distributed angles
- Uses torchvision transforms
- Maintains spatial relationships

**Class Split Tasks:**

- Divides classes evenly across tasks
- Filters data by class membership
- Re-writes all labels to a 1 or 2


### Comprehensive Visualization

The system automatically generates analysis pages and files.
The unused visualizations are currently disabled and broken. Enable at your own risk.

**Generated Files:**

- `test_results/ta_agem_metrics_YYYYMMDD_HHMMSS.pkl`: Raw metrics data
- You can use `read_pickle.py` to open these files one at a time, it will prompt you for the location.
- You can use `visualization_analysis/retro_stats.py -n 15 --input_dir PATH_TO_PICKLE_FILES_DIRECTORY` for checking the statistics of $n$ runs in a particular folder.
- You can use `visualization_analysis/compare_experiments.py PATH_TO_DIR1 PATH_TO_DIR2 ... PATH_TO_DIRN` for comparing folders of experiments.

### Key Metrics Tracked

**Performance Metrics:**

- Per-task accuracy after each training phase
- Average accuracy across all seen tasks
- Forgetting rates between tasks
- Training time per task

**Memory Metrics: (UNUSED)**

- Total samples stored across all pools
- Samples per pool (class-wise distribution)
- Number of active pools
- Memory efficiency ratios

## Experimental Results

### Expected Behavior

**Successful Run Indicators:**

- Accuracy should remain stable or improve across tasks
- Memory size should grow bounded by Q×P×num_pools
- Training time should remain consistent per task
- Forgetting should be minimal compared to naive approaches

### Performance Tuning

**For Better Accuracy:**

- Increase `MEMORY_SIZE_Q` (more clusters per pool)
- Increase `MEMORY_SIZE_P` (more samples per cluster)
- Reduce `LEARNING_RATE` for more stable training

**For Faster Training:**

- Reduce `BATCH_SIZE` for more frequent updates
- Reduce `TESTING_INTERVAL` for less frequent tests
- Enable `QUICK_TEST_MODE` for development

**For Memory Efficiency:**

- Reduce `MEMORY_SIZE_Q` and `MEMORY_SIZE_P`
- Monitor pool sizes to avoid imbalanced memory

## Troubleshooting

### Common Issues

**CUDA/GPU Issues:**

```bash
# Force CPU-only execution
export CUDA_VISIBLE_DEVICES=""
python main.py
```

**Memory Errors:**

- Reduce `BATCH_SIZE` or `MEMORY_SIZE_Q`
- Enable `QUICK_TEST_MODE`
- Monitor system memory usage

**Dataset Download Issues:**

- Check internet connection
- Manually download dataset
- Pass in the location of the dataset with `--data_dir` - like `python main.py --data_dir ./datasets`
  - The dataset should be in a subfoler inside the path you pass in, and named after the relevant dataset in uppercase, like `MNIST`

**Import Errors:**

- Verify conda environment activation
- Install missing dependencies manually

### Debug Mode

Use VS code debugging. You will need to select the correct python interpreter in the conda environment.

### Validation

**Quick Validation Test:**

```python
# Run with minimal configuration
python3 main.py --lite
```

## Algorithm Details

### Default TA-A-GEM Algorithm Flow

1. **Initialization:**

   - Create SimpleMLP model
   - Initialize A-GEM handler for gradient projection
   - Create ClusteringMemory with multi-pool architecture

2. **For each task:**

   - **For each epoch:**
     - **For each batch:**
       - Compute A-GEM gradient projection using memory
       - Update model parameters
       - Add 1 sample to appropriate clustering pool
   - Evaluate on all seen tasks
   - Update performance metrics

3. **Post-training:**
   - Save metrics

### Memory Management Strategy

**Pool Assignment:**

- 1 pool per class

**Clustering Within Pools:**

- Maximum Q clusters per pool
- Maximum P samples per cluster

**Memory Retrieval:**

- Samples retrieved from all active pools
- Used for gradient projection in A-GEM
- Maintains class balance across tasks

## Extensions and Customization

### Adding New Task Types

Extend `prepare_domain_incremental_mnist` in `mnist.py`:

```python
elif task_type == 'your_new_task':
    # Implement your task transformation
    for task_id in range(num_tasks):
        # Transform data for this task
        # Create DataLoader
        task_dataloaders.append(dataloader)
```

### Custom Neural Networks

Replace `SimpleMLP` in `main.py`:

```python
class YourCustomModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        # Your architecture

    def forward(self, x):
        # Your forward pass
        return x
```

### Different Datasets

Implement new data loading using abstract classes in `load_dataset.py`

## MLflow Integration

This project includes MLflow integration for experiment tracking and visualization.

### Prerequisites

Install MLflow:
```bash
pip install mlflow
# If you encounter protobuf errors:
pip install "protobuf<4.0.0"
```

### Usage

**New Experiments (Automatic):**
The system automatically logs to MLflow when running experiments:

```bash
# Single experiment
python main.py --task_type permutation --experiment_name my-test

# Batch experiments
./run_experiments.sh mnist my-experiment-batch
```

**Importing Existing Results:**
Import old pickle files into MLflow. Will simulate the run, so takes as long as the run did.

```bash
# Import last 5 results from a folder
python utils/import_to_mlflow.py --input_dir test_results/my-old-experiment --last 5

# Import all results from a folder
python utils/import_to_mlflow.py --input_dir test_results/my-old-experiment --all

# Dry run to see what would be imported
python utils/import_to_mlflow.py --input_dir test_results/my-old-experiment --last 10 --dry_run
```

**Viewing Results:**
```bash
mlflow ui
# Open http://localhost:5000 in your browser
```

### MLflow Features

**Experiment Organization:**
- Experiments named after your folder structure
- Runs automatically named with task type and timestamp
- Tags for easy filtering (task_type, dataset, quick_test)

**Metrics Tracked:**
- Per-TESTING_RATE-batches: overall_accuracy, loss, memory_size, learning_rate
- Per-task: individual task accuracies (task_0_accuracy, task_1_accuracy, etc.)
- Batch-level: batch_loss (sampled every 10 batches)
- Summary: final_accuracy, average_accuracy

**Artifacts Stored:**
- Original pickle files
- HTML visualization graphs
- Model checkpoints (if enabled)

**Comparison Features:**
- Compare runs side by side
- Filter by parameters or performance
- Download artifacts and visualizations
- Export data for further analysis

## Acknowledgments

This work is based on the original A-GEM[1] and TA-A-GEM[2] paper, which can be found here:

- [A-GEM Paper](https://doi.org/10.48550/arXiv.1812.00420)
  [1] A. Chaudhry, M. Ranzato, M. Rohrbach, and M. Elhoseiny, “Efficient lifelong learning with a-gem,” arXiv preprint arXiv:1812.00420, 2018.
- [TA-A-GEM Paper](https://doi.org/10.48550/arXiv.2309.12078)
  [2] C. Lamers, R. Vidal, N. Belbachir, N. van Stein, T. Bäeck, and P. Giampouras, “Clustering-based domain-incremental learning,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 3384–3392.

## License

This implementation is provided for research purposes. Please refer to the original A-GEM and clustering algorithm papers for their respective licenses.

## Contact

For questions, issues, or contributions, please open an issue in the repository.
