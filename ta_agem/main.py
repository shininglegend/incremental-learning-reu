# Code written by Gemini 2.5 Flash.
# Edited by Titus Murphy and Abigail Dodd
import agem
import clustering
import mnist
# import torch
import torch.nn as nn
import torch.optim as optim
# import numpy as np
# import pandas as pd
from visualization_analysis import TAGemVisualizer
from parallel_ta_agem import ParallelTrainingManager, create_parallel_params, \
    run_parallel_ta_agem, run_enhanced_parallel_ta_agem
from simple_parallel_ta_agem import SimpleParallelTrainingManager, run_simple_parallel_ta_agem
import time
from datetime import datetime
import math
import pickle
from evaluation import TAGEMEvaluator, quick_evaluate_model
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


# --- Configuration and Initialization ---
# If set to True, will w run fewer tasks and fewer data, and logs loss per batch
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
TASK_TYPE = 'permutation'  # 'permutation', 'rotation', or 'class_split'.

# Parallel processing configurations. See parallel_ta_agem for details
ENABLE_PARALLEL = True  # Set to False to use original sequential training
USE_SIMPLE_PARALLEL = False  # Set to True to use threading instead of multiprocessing
PARALLEL_STRATEGY = 'memory_update'  # Options: 'memory_update', 'pool_training', 'batch_processing', 'hybrid'
NUM_PROCESSES = None  # None to use all available cores, or specify a number
NUM_THREADS = 4  # Number of threads for simple parallel approach

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
    'quick_test_mode': QUICK_TEST_MODE,
    'enable_parallel': ENABLE_PARALLEL,
    'use_simple_parallel': USE_SIMPLE_PARALLEL,
    'parallel_strategy': PARALLEL_STRATEGY,
    'num_processes': NUM_PROCESSES,
    'num_threads': NUM_THREADS
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


def print_program_runtime():
    run_time = time.time() - program_start_time
    run_time_minutes = math.floor(run_time / 60)
    run_time_seconds = run_time - run_time_minutes*60
    print(f"It took {run_time_minutes}min and {run_time_seconds:.2f}s.")


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
# It returns a list of data loaders, one for each task/domain
task_dataloaders = mnist.prepare_domain_incremental_mnist(
    task_type=TASK_TYPE, num_tasks=NUM_TASKS, batch_size=BATCH_SIZE,
    quick_test=QUICK_TEST_MODE
)

# --- Initialize comprehensive visualizer ---
visualizer = TAGemVisualizer()

# --- Timing ---
program_start_time = time.time()
print(f"The program started at {datetime.now().strftime('%H:%M:%S')}")

# --- Training Loop ---
print("Starting TA-A-GEM training...")
if ENABLE_PARALLEL:
    if USE_SIMPLE_PARALLEL:
        # Use simple threading-based parallel implementation
        print(f"Using simple parallel training with {NUM_THREADS} threads")
        run_simple_parallel_ta_agem(
            params, model, optimizer, criterion, clustering_memory,
            task_dataloaders, agem_handler, visualizer
        )
    else:
        # Use enhanced multiprocessing-based parallel implementation
        print(f"Using enhanced parallel training with strategy: {PARALLEL_STRATEGY}")

        # Add timeout and progress tracking parameters
        params['timeout_seconds'] = 1000  # Seconds for timeout per task
        params['show_progress'] = True

        try:
            run_enhanced_parallel_ta_agem(
                params, model, optimizer, criterion, clustering_memory,
                task_dataloaders, agem_handler, visualizer
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Partial results may be available.")
        except Exception as e:
            print(f"\nParallel training failed: {e}")
            print("Falling back to sequential training...")

            # Fallback to sequential training
            ENABLE_PARALLEL = False  # Disable parallel for fallback
            # Continue with the original sequential training code below
else:
    # Sequential training loop
    for task_id, task_dataloader in enumerate(task_dataloaders):
        print(f"\n--- Training on Task {task_id} ---")

        task_start_time = time.time()
        task_epoch_losses = []

        for epoch in range(NUM_EPOCHS):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (data, labels) in enumerate(task_dataloader):
                batch_loss = agem_handler.optimize(data, labels, clustering_memory.get_memory_samples())

                if batch_loss is not None:
                    epoch_loss += batch_loss
                    visualizer.add_batch_loss(task_id, epoch, batch_idx, batch_loss)

                for i in range(len(data)):
                    sample_data = data[i]
                    sample_label = labels[i]
                    clustering_memory.add_sample(sample_data, sample_label)

                num_batches += 1

                # Progress bar logic
                if not QUICK_TEST_MODE and (batch_idx % 50 == 0 or batch_idx == len(task_dataloader) - 1):
                    progress = (batch_idx + 1) / len(task_dataloader)
                    bar_length = 30
                    filled_length = int(bar_length * progress)
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)
                    print(f'\rTask {task_id:1}, Epoch {epoch+1:>2}/{NUM_EPOCHS}: |{bar}| {progress:.1%} ({batch_idx + 1}/{len(task_dataloader)})', end='', flush=True)

            if not QUICK_TEST_MODE: print()

            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            task_epoch_losses.append(avg_epoch_loss)

            if QUICK_TEST_MODE and (epoch % 5 == 0 or epoch == NUM_EPOCHS - 1):
                print(f'  Epoch {epoch+1}/{NUM_EPOCHS}: Loss = {avg_epoch_loss:.4f}')

        # Evaluation and metrics
        model.eval()
        avg_accuracy = agem.evaluate_tasks_up_to(model, criterion, task_dataloaders, task_id)

        individual_accuracies = []
        for eval_task_id in range(task_id + 1):
            eval_dataloader = task_dataloaders[eval_task_id]
            task_acc = agem.evaluate_single_task(model, criterion, eval_dataloader)
            individual_accuracies.append(task_acc)

        task_time = time.time() - task_start_time

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
print_program_runtime()

# Pause after training is done to see if user wants to test.
while True:
    user_input = input("Enter 'y' to evaluate on test data, or 'n' to exit: ").strip().lower()
    if user_input == 'y':
        break
    elif user_input == 'n':
        exit("if you say so....")
    else:
        print("Can you read? Try 'y' or 'n', not whatever you just did. ")
print("Let's evaluate this hoe.\n")


# --- COMPREHENSIVE TEST EVALUATION ---
print("\n" + "="*60)
print("STARTING TEST DATA EVALUATION")
print("="*60)

# Create evaluator and run comprehensive evaluation
evaluator = TAGEMEvaluator(params, save_dir="./test_results")
test_results = evaluator.run_full_evaluation(model, device='cpu', save_results=True)

# --- Optional: Quick comparison with baseline ---
# You can uncomment this if you want to compare with a baseline model
"""
# Create a baseline model (randomly initialized) for comparison
baseline_model = SimpleMLP(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES)
from evaluation import compare_models

comparison_results = compare_models(
    models=[model, baseline_model],
    model_names=['TA-AGEM', 'Baseline'],
    params=params
)
"""

# --- Generate final summary ---
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Training and testing completed in: ", end="")
print_program_runtime()
print(f"Final test accuracy: {test_results['overall_accuracy']:.4f}")
print(f"Results saved with timestamp: {evaluator.timestamp}")

# --- Optional: Save training history alongside test results ---
final_results = {
    'training_params': params,
    'test_results': test_results,
    'training_visualizer_data': visualizer.get_all_metrics() if 'visualizer' in locals() else None,
    'model_state_dict': model.state_dict()
}

final_results_path = f"./test_results/complete_results_{evaluator.timestamp}.pkl"
with open(final_results_path, 'wb') as f:
    pickle.dump(final_results, f)
print(f"Complete results saved to: {final_results_path}")
