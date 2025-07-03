# TA-A-GEM: Topology-Aware Averaged Gradient Episodic Memory

This folder contains an implementation of **Topology-Aware Averaged Gradient Episodic Memory (TA-A-GEM)**, a continual learning approach that combines A-GEM's gradient projection with topology-aware clustering for episodic memory management. It is based on the paper by Lamers et al. (2023).

## Overview

TA-A-GEM addresses catastrophic forgetting in continual learning by:
- Using **gradient projection** (A-GEM) to prevent interference between tasks
- Implementing **topology-aware clustering** to maintain representative samples in episodic memory
- Supporting **multi-pool architecture** for different task types
- Providing comprehensive **visualization and analysis** tools

## Architecture

The implementation consists of several key components:

### Core Components
- **`main.py`**: Main training loop and experiment orchestration
- **`agem.py`**: A-GEM gradient projection implementation
- **`clustering.py`**: Multi-pool clustering memory management
    - **Multi-pool management**: Separate clustering pools per class/task type
- **`load_dataset.py`**: Abstract data loading and task generation
- **`visualization_analysis.py`**: Comprehensive analysis and visualization tools
- **`clustering_gemini.py`** for core clustering algorithms

### Dependencies
- See the \*_env.yml files for dependencies.

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

### Dataset Setup
The code automatically downloads the datasets via kagglehub. If you encounter issues:
1. Manually download the correct dataset from: https://www.kaggle.com/
2. Update the `path` variable in the relevant dataset file with your path.

## Usage

### Basic Execution

**Quick Test Mode (Recommended for initial testing):**
```bash
# Edit main.py and set QUICK_TEST_MODE = True
python3 main.py
```
- Runs 2 tasks with reduced dataset
- 20 epochs per task
- Detailed loss logging per batch
- Faster execution for debugging

**Full Experiment Mode:**
```bash
# Edit main.py and set QUICK_TEST_MODE = False
python3 main.py
```
- Runs 5 tasks with full dataset
- 20 epochs per task
- Progress bar display
- Complete experimental setup

### Configuration Parameters

Key parameters can be modified in `main.py`:

#### Task Configuration
```python
NUM_TASKS = 2 if QUICK_TEST_MODE else 5  # Number of continual learning tasks
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
NUM_POOLS = 10       # Number of memory pools (tied to task_type)
```

#### Training Parameters
```python
BATCH_SIZE = 10      # Batch size (as per paper)
LEARNING_RATE = 1e-3 # Learning rate
NUM_EPOCHS = 20      # Epochs per task
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

The `ClusteringMemory` class manages topology-aware episodic memory:

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
- Generates unique random permutations
- Applies to flattened dataset images
- Preserves original labels

**Rotation Tasks:**
- Calculates evenly distributed angles
- Uses torchvision transforms
- Maintains spatial relationships

**Class Split Tasks:**
- Divides classes evenly across tasks
- Filters data by class membership
- Preserves original class labels

## Monitoring and Analysis

### Real-time Progress

**Quick Test Mode:**
```
--- Training on Task 0 ---
  Epoch 1/20: Loss = 2.1234
  Epoch 5/20: Loss = 1.8765
  ...
After Task 0, Average Accuracy: 0.8567
Memory Size: 450 samples across 8 active pools
```

**Full Mode:**
```
Task 0, Epoch  1/20: |██████████████████████████████| 100.0% (600/600)
After Task 0, Average Accuracy: 0.8567
Memory Size: 450 samples across 8 active pools
Pool sizes: (should be maxed at P)
```

### Comprehensive Visualization

The system automatically generates analysis pages and files.
The unused visualizations are currently disabled and broken. Enable at your own risk.

**Generated Files:**
- `test_results/ta_agem_metrics_YYYYMMDD_HHMMSS.pkl`: Raw metrics data
- (UNUSED) `test_results/ta_agem_analysis_YYYYMMDD_HHMMSS.html`: Interactive dashboard
- (UNUSED) `test_results/accuracy_progression_YYYYMMDD_HHMMSS.png`: Accuracy plots
- (UNUSED) `test_results/memory_analysis_YYYYMMDD_HHMMSS.png`: Memory usage plots

**Generated Pages:**
- Storage visualization in each pool of clusters
- Per-task and overall accuracy as a function of time

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

**Typical Results:**
```
Task 0: Accuracy = 0.93-0.97 (baseline performance)
Task 0: Accuracy = 0.92-0.96 (usually slightly worse performance)
Task 0: Accuracy = 0.91-0.95 (might stabilize)
...
```

### Performance Tuning

**For Better Accuracy:**
- Increase `MEMORY_SIZE_Q` (more clusters per pool)
- Increase `MEMORY_SIZE_P` (more samples per cluster)
- Reduce `LEARNING_RATE` for more stable training
- Increase `NUM_EPOCHS` for better task learning

**For Faster Training:**
- Reduce `BATCH_SIZE` for more frequent updates
- Enable `QUICK_TEST_MODE` for development
- Reduce `NUM_EPOCHS` for quicker iterations

**For Memory Efficiency:**
- Reduce `MEMORY_SIZE_Q` and `MEMORY_SIZE_P`
- Use `class_split` tasks (fewer pools needed)
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
- Update `path` variable

**Import Errors:**
- Verify conda environment activation
- Install missing dependencies manually

### Debug Mode

Use VS code debugging. You will need to select the correct python interpreter in the conda environment.

### Validation

**Quick Validation Test:**
```python
# Run with minimal configuration
QUICK_TEST_MODE = True
```

## Algorithm Details

### TA-A-GEM Algorithm Flow

1. **Initialization:**
   - Create SimpleMLP model
   - Initialize AGEMHandler for gradient projection
   - Create ClusteringMemory with multi-pool architecture

2. **For each task:**
   - **For each epoch:**
     - **For each batch:**
       - Compute A-GEM gradient projection using memory
       - Update model parameters
       - Add samples to appropriate clustering pools
   - Evaluate on all seen tasks
   - Update performance metrics

3. **Post-training:**
   - Generate analysis
   - Save metrics

### Memory Management Strategy

**Pool Assignment:**
- Permutation/Rotation: 10 pools (one per class)
- Class Split: 2 pools (task-based assignment)

**Clustering Within Pools:**
- Maximum Q clusters per pool
- Maximum P samples per cluster
- Topology-aware sample selection

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

## Acknowledgments

This work is based on the original A-GEM[1] and TA-A-GEM[2] paper, which can be found here:
- [A-GEM Paper](https://doi.org/10.48550/arXiv.1812.00420) 
[1] A. Chaudhry, M. Ranzato, M. Rohrbach, and M. Elhoseiny, “Efficient lifelong learning with a-gem,” arXiv preprint arXiv:1812.00420, 2018.
- [TA-A-GEM Paper](https://doi.org/10.48550/arXiv.2309.12078)
[2] C. Lamers, R. Vidal, N. Belbachir, N. van Stein, T. Bäeck, and P. Giampouras, “Clustering-based domain-incremental learning,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 3384–3392.

## License

This implementation is provided for research purposes. Please refer to the original A-GEM and clustering algorithm papers for their respective licenses.

## Contact

For questions, issues, or contributions, please  open an issue in the repository.