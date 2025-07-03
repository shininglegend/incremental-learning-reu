# Code written by Gemini 2.5 Flash. Edited by Titus
import agem
import clustering
import mnist
# import torch
import torch.nn as nn
import torch.optim as optim
# import numpy as np
# import pandas as pd
from visualization_analysis import TAGemVisualizer
import time

# --- 1. Configuration and Initialization ---
# If set to True, will w run less tasks and less data, and logs loss per batch
# If set to False, will run full MNIST with 5 tasks and 10 epochs with normal progress bar
QUICK_TEST_MODE = False

NUM_CLASSES = 10 # For MNIST
INPUT_DIM = 784  # For MNIST (28*28)
HIDDEN_DIM = 200 # As per paper
MEMORY_SIZE_Q = 10 # Number of clusters per pool (Q)
MEMORY_SIZE_P = 3  # Max samples per cluster (P) - as per paper
BATCH_SIZE = 50 if QUICK_TEST_MODE else 10    # As per paper
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
NUM_TASKS = 2 if QUICK_TEST_MODE else 5 # Example: for permutation or rotation based tasks
TASK_TYPE = 'permutation'  # 'permutation', 'rotation', or 'class_split'

# Determine number of pools based on task type (as per paper)
if TASK_TYPE in ['permutation', 'rotation']:
    NUM_POOLS = 10  # 10 pools for permutation/rotation tasks
    CLUSTERS_PER_POOL = 10  # Q = 10 per pool
elif TASK_TYPE == 'class_split':
    NUM_POOLS = 2   # 2 pools for class split tasks
    CLUSTERS_PER_POOL = 50  # Q = 50 per pool
else:
    NUM_POOLS = NUM_CLASSES  # Default fallback
    CLUSTERS_PER_POOL = MEMORY_SIZE_Q

params = {  # Store all parameters for easy access
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
    'quick_test_mode': QUICK_TEST_MODE
}

# Define a simple MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Ensure consistent flattening regardless of input shape
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)  # Flatten to (batch_size, features)
        elif len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Initialize model, optimizer, and loss function
model = SimpleMLP(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Initialize TA-A-GEM components with multi-pool architecture
# Clustering_memory will manage the episodic memory with separate pools per class
clustering_memory = clustering.ClusteringMemory(
    Q=CLUSTERS_PER_POOL, P=MEMORY_SIZE_P, input_type='samples', num_pools=NUM_POOLS
)

# A-GEM wrapper/class for gradient projection logic
# It needs to access the memory managed by clustering_memory
agem_handler = agem.AGEMHandler(model, criterion, optimizer)

# Load and prepare MNIST data for domain-incremental learning
# This function would encapsulate the permutation, rotation, or class split logic
# It returns lists of train and test data loaders, one for each task/domain
train_dataloaders, test_dataloaders = mnist.prepare_domain_incremental_mnist(
    task_type=TASK_TYPE, num_tasks=NUM_TASKS, batch_size=BATCH_SIZE,
    quick_test=QUICK_TEST_MODE
)

# --- Initialize comprehensive visualizer ---
visualizer = TAGemVisualizer()

# --- 2. Training Loop ---
print("Starting TA-A-GEM training...")
for task_id, train_dataloader in enumerate(train_dataloaders):
    print(f"\n--- Training on Task {task_id} ---")

    task_start_time = time.time()
    task_epoch_losses = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (data, labels) in enumerate(train_dataloader):
            # Step 1: Use A-GEM logic for current batch and current memory
            # agem_handler.optimize handles model update and gradient projection
            # It queries clustering_memory for the current reference samples
            batch_loss = agem_handler.optimize(data, labels, clustering_memory.get_memory_samples())

            # Track batch loss
            if batch_loss is not None:
                epoch_loss += batch_loss
                visualizer.add_batch_loss(task_id, epoch, batch_idx, batch_loss)



            # Step 2: Update the clustered memory with current batch samples
            # This is where the core clustering for TA-A-GEM happens
            for i in range(len(data)):
                sample_data = data[i]
                sample_label = labels[i]
                clustering_memory.add_sample(sample_data, sample_label) # Add sample to clusters

            num_batches += 1

            # Update progress bar every 50 batches or on last batch
            if not QUICK_TEST_MODE and (batch_idx % 50 == 0 or batch_idx == len(train_dataloader) - 1):
                progress = (batch_idx + 1) / len(train_dataloader)
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f'\rTask {task_id:1}, Epoch {epoch+1:>2}/{NUM_EPOCHS}: |{bar}| {progress:.1%} ({batch_idx + 1}/{len(train_dataloader)})', end='', flush=True)

        # Print newline after progress bar completion
        if not QUICK_TEST_MODE: print()

        # Track epoch loss
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        task_epoch_losses.append(avg_epoch_loss)

        # Print epoch summary
        if QUICK_TEST_MODE and (epoch % 5 == 0 or epoch == NUM_EPOCHS - 1):
            print(f'  Epoch {epoch+1}/{NUM_EPOCHS}: Loss = {avg_epoch_loss:.4f}')

    # Evaluate performance after each task on TEST DATA
    model.eval()
    avg_accuracy = agem.evaluate_tasks_up_to(model, criterion, test_dataloaders, task_id)

    # Evaluate on individual tasks for detailed tracking
    individual_accuracies = []
    for eval_task_id in range(task_id + 1):
        eval_dataloader = test_dataloaders[eval_task_id]
        task_acc = agem.evaluate_single_task(model, criterion, eval_dataloader)
        individual_accuracies.append(task_acc)

    # Calculate training time for this task
    task_time = time.time() - task_start_time

    # Update visualizer with all metrics
    memory_size = clustering_memory.get_memory_size()
    pool_sizes = clustering_memory.get_pool_sizes()
    num_active_pools = clustering_memory.get_num_active_pools()

    visualizer.update_metrics(
        task_id=task_id,
        overall_accuracy=avg_accuracy,
        individual_accuracies=individual_accuracies,
        epoch_losses=task_epoch_losses,
        memory_size=memory_size,
        training_time=task_time
    )

    print(f"After Task {task_id}, Average Accuracy: {avg_accuracy:.4f}")
    print(f"Memory Size: {memory_size} samples across {num_active_pools} active pools")
    print(f"Pool sizes: {pool_sizes}")
    print(f"Task Training Time: {task_time:.2f}s")

print("\nTA-A-GEM training complete.")

# --- 3. Comprehensive Visualization and Analysis ---
print("\nGenerating comprehensive analysis...")

# Save metrics for future analysis
timestamp = time.strftime("%Y%m%d_%H%M%S")
visualizer.save_metrics(f"test_results/ta_agem_metrics_{timestamp}.pkl", params=params)

# Generate simplified report with 3 key visualizations
visualizer.generate_simple_report(clustering_memory)

print(f"\nAnalysis complete! Files saved with timestamp: {timestamp}")
