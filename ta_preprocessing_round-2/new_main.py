import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import pickle
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Import existing modules
import agem
import clustering
import mnist
from visualization_analysis import TAGemVisualizer
from evaluation import TAGEMEvaluator
from load_dataset import load_dataset
from simple_mlp import SimpleMLP
import processing
from advanced_parallel_strategies import _run_hybrid_training, run_ensemble_parallel_training # Import _run_hybrid_training

# --- CUDA Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
# ---------------------------------

# Main function
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # Configuration
    import config
    params = config.get_params()

    print(f"Program started at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Configuration: {params['num_tasks']} tasks, {params['num_epochs']} epochs")

    # Load data
    print("Loading dataset and preparing data loaders...")
    datasetLoader = load_dataset(params['dataset_name'])
    train_dataloaders, test_dataloaders, permutations_for_tasks = datasetLoader.prepare_domain_incremental_data(
        task_type=params['task_type'], num_tasks=params['num_tasks'], batch_size=params['batch_size'],
        quick_test=params['quick_test_mode']
    )

    print(f"\nData Overview:")
    for i, dataloader in enumerate(train_dataloaders): # Changed task_dataloaders to train_dataloaders
        print(f"  Task {i}: {len(dataloader)} batches")

    choice = params['strategy']

    model = None # Initialize model outside the if-else block
    memory = None
    results = None
    training_time = 0

    if choice == '1':
        model, memory, results, training_time = processing.run_optimized_training(params, train_dataloaders)
    elif choice == '2':
        # For hybrid, we need an initial model instance to pass
        model = SimpleMLP(params['input_dim'], params['hidden_dim'], params['num_classes']).to(DEVICE)
        model, memory, results, training_time = _run_hybrid_training(params, model, train_dataloaders) # Pass train_dataloaders
    else:
        model, memory, training_time = processing.run_sequential_training(params, train_dataloaders) # Pass train_dataloaders
        results = None

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETED!")
    print(f"Total training time: {training_time // 60:.0f}m {training_time % 60:.0f}s")
    print(f"{'=' * 60}")

    # Evaluation
    print("\nStarting evaluation...")

    if params['task_type'] == 'permutation' and permutations_for_tasks is not None:
        params['permutations'] = permutations_for_tasks
    else:
        # If it's not a permutation task, or permutations are unexpectedly None,
        # ensure 'permutations' key is absent or set to None in params
        params['permutations'] = None


    if model is not None: # Ensure model exists before evaluation
        evaluator = TAGEMEvaluator(params, test_dataloaders=test_dataloaders, save_dir="./test_results")
        test_results = evaluator.run_full_evaluation(model, device='cpu', save_results=True)

        print(f"Final test accuracy: {test_results['overall_accuracy']:.4f}")
    else:
        print("No model to evaluate. Training might have failed or not produced a model.")


    # Set multiprocessing start method
    torch.autograd.set_detect_anomaly(True)
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
