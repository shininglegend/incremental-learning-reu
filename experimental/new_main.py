import torch
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Import existing modules
from evaluation import TAGEMEvaluator
from load_dataset import load_dataset
from simple_mlp import SimpleMLP
import processing
import load_dataset
from advanced_parallel_strategies import _run_hybrid_training  # Import _run_hybrid_training
from config import params, parse_arguments
DEVICE = params['device']


# Main function
def main():
    mp.set_start_method('spawn', force=True)

    # --- CUDA Device Configuration ---
    print(f"Using device: {DEVICE}")
    # ---------------------------------

    # Configuration
    import config
    params = config.get_params()

    print(f"Program started at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Configuration: {params['num_tasks']} tasks, {params['num_epochs']} epochs")

    # Load data
    print(f"Task type: {params['task_type']}")
    print("Loading dataset and preparing data loaders...")
    train_dataloaders, test_dataloaders = load_dataset.prepare_domain_incremental_data(
        dataset_name=params['dataset_name'],
        task_type=params['task_type'],
        num_tasks=params['num_tasks'],
        batch_size=params['batch_size'],
        quick_test=params['quick_test_mode']
    )

    print(f"\nData Overview:")
    for i, dataloader in enumerate(train_dataloaders):
        print(f"  Task {i}: {len(dataloader)} batches")

    choice = params['strategy']

    model = None  # Initialize model outside the if-else block
    memory = None
    results = None
    training_time = 0
    intermediate_eval_accuracies_history = []

    if choice == '1':
        model, memory, results, training_time, intermediate_eval_accuracies_history = \
            processing.run_optimized_training(params, train_dataloaders, test_dataloaders)
    elif choice == '2':
        # For hybrid, we need an initial model instance to pass
        model = SimpleMLP(params['input_dim'], params['hidden_dim'], params['num_classes']).to(DEVICE)
        model, memory, results, training_time, intermediate_eval_accuracies_history = _run_hybrid_training(
            params, model, train_dataloaders)
    else:
        model, memory, training_time, intermediate_eval_accuracies_history = processing.run_sequential_training(
            params, train_dataloaders, test_dataloaders)
        results = None

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETED!")
    print(f"Total training time: {training_time // 60:.0f}m {training_time % 60:.0f}s")
    print(f"{'=' * 60}")

    # Evaluation
    print("\nStarting evaluation...")

    if model is not None:  # Ensure model exists before evaluation
        evaluator = TAGEMEvaluator(test_dataloaders=test_dataloaders, save_dir=params["output_dir"])
        test_results = evaluator.run_full_evaluation(model, device='cpu', save_results=True,
                                                     intermediate_eval_history=intermediate_eval_accuracies_history)

        print(f"Final test accuracy: {test_results['overall_accuracy']:.4f}")
    else:
        print("No model to evaluate. Training might have failed or not produced a model.")


if __name__ == '__main__':
    main()
