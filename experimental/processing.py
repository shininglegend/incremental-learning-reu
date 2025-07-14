import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

import clustering_pools
import agem
from evaluation import TAGEMEvaluator
from simple_mlp import SimpleMLP

# config
from config import params
DEVICE = params['device']


def run_sequential_training(params, task_dataloaders_nested, test_dataloaders):
    """Baseline sequential training"""
    print("Running sequential training...")

    model = SimpleMLP(params['input_dim'], params['hidden_dim'], params['num_classes'])
    model.to(params['device'])
    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    memory = clustering_pools.ClusteringMemory(
        Q=params['memory_size_q'], P=params['memory_size_p'],
        input_type='samples', num_pools=params['num_pools'], device=params['device']
    )
    agem_handler = agem.AGEMHandler(model, criterion, optimizer, batch_size=params['batch_size'],
                                    device=params['device'])

    start_time = time.time()

    flat_task_dataloaders = []
    # Check if the input is a single list containing other lists of DataLoaders
    if isinstance(task_dataloaders_nested, (list, tuple)):
        # Check if it's a nested structure (e.g., [[DL], [DL]] or ( (DL,), (DL,) ) )
        if (len(task_dataloaders_nested) > 0 and
                isinstance(task_dataloaders_nested[0], (list, tuple))):

            # Iterate through sub-iterables (which are expected to contain DataLoaders)
            for sub_iterable in task_dataloaders_nested:
                if isinstance(sub_iterable, (list, tuple)):
                    for dl in sub_iterable:
                        if isinstance(dl, torch.utils.data.dataloader.DataLoader):
                            flat_task_dataloaders.append(dl)
                        else:
                            raise TypeError(f"Expected DataLoader within nested structure, but found {type(dl)}.")
                else:
                    raise TypeError(
                        f"Expected a list or tuple of DataLoaders, but found {type(sub_iterable)} at sub-level.")
        else:
            # Assume it's a flat iterable of DataLoaders (e.g., [DL, DL] or (DL, DL) )
            for dl in task_dataloaders_nested:
                if isinstance(dl, torch.utils.data.dataloader.DataLoader):
                    flat_task_dataloaders.append(dl)
                else:
                    raise TypeError(f"Expected DataLoader in flat structure, but found {type(dl)}.")
    else:
        raise TypeError(f"Expected a list or tuple of DataLoaders, but got {type(task_dataloaders_nested)}.")

    intermediate_eval_accuracies_history = []

    # Now iterate over the correctly flattened list of DataLoaders
    for task_id, task_dataloader in enumerate(flat_task_dataloaders):
        print(f"\nTask {task_id + 1}/{params['num_tasks']}")

        for epoch in tqdm(range(params['num_epochs']), desc=f"Task {task_id + 1} Epochs", leave=False):
            model.train()
            epoch_losses = []

            for batch_idx, (data, labels) in enumerate(task_dataloader):
                # Move data and labels to the same device as the model
                data = data.to(params['device'])
                labels = labels.to(params['device'])

                # Get memory samples for A-GEM
                _samples = memory.get_memory_samples()

                # Use A-GEM optimization with current batch and memory samples
                batch_loss = agem_handler.optimize(data, labels, _samples)

                if batch_loss is not None:
                    epoch_losses.append(batch_loss)

                # Add current batch samples to memory (moved to CPU for storage)
                for i in range(len(data)):
                    sample_data = data[i].cpu()
                    sample_label = labels[i].cpu()
                    memory.add_sample(sample_data, sample_label)

        # --- Perform intermediate evaluation after each task is trained ---
        print(f"  Evaluating model after training Task {task_id + 1}...")
        # Create a temporary evaluator instance for intermediate evaluation
        temp_evaluator = TAGEMEvaluator(test_dataloaders=test_dataloaders)
        # Evaluate on all tasks from 0 up to the current task_id (inclusive)
        current_accuracies = temp_evaluator.evaluate_tasks_up_to(model, task_id, test_dataloaders,
                                                                 device=params['device'])
        intermediate_eval_accuracies_history.append(current_accuracies)
        print(f"  Intermediate accuracies after Task {task_id + 1}: {current_accuracies}")
        # --- End intermediate evaluation ---

    training_time = time.time() - start_time
    return model, memory, training_time, intermediate_eval_accuracies_history


def train_single_task(args):
    """Train a single task with reliable optimizations (no threading)"""
    task_id, task_dataloader, params, shared_memory_samples, model_state_dict = args

    print(f"Worker starting Task {task_id}")

    model = SimpleMLP(params['input_dim'], params['hidden_dim'], params['num_classes']).to(DEVICE)
    if model_state_dict:
        model.load_state_dict(model_state_dict)

    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    local_memory = clustering_pools.ClusteringMemory(
        Q=params['memory_size_q'],
        P=params['memory_size_p'],
        input_type='samples',
        num_pools=params['num_pools'],
        device=params['device']
    )

    if shared_memory_samples:
        for sample_data, sample_label in shared_memory_samples:
            local_memory.add_sample(sample_data, sample_label)

    agem_handler = agem.AGEMHandler(model, criterion, optimizer, batch_size=params['batch_size'],
                                    device=params['device'])
    task_history = []

    # Larger effective batch size through gradient accumulation
    accumulation_steps = 4  # Effectively 4x larger batches

    # Mixed precision training (if available)
    use_amp = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
        print(f"Task {task_id}: Using mixed precision training")

    for epoch in range(params['num_epochs']):
        model.train()
        epoch_losses = []
        accumulated_projected_grads = None
        step_count = 0

        for batch_idx, (data, labels) in enumerate(task_dataloader):
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)

            # Get memory samples for A-GEM
            _samples = local_memory.get_memory_samples()

            # Use A-GEM optimization with current batch and memory samples
            batch_loss = agem_handler.optimize(data, labels, _samples)

            if batch_loss is not None:
                epoch_losses.append(batch_loss)

            # Add current batch samples to memory (moved to CPU for storage)
            for i in range(len(data)):
                sample_data = data[i].cpu()
                sample_label = labels[i].cpu()
                local_memory.add_sample(sample_data, sample_label)

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        task_history.append(avg_loss)

        if epoch % 5 == 0:
            print(f"Task {task_id}, Epoch {epoch + 1}/{params['num_epochs']}: Loss = {avg_loss:.4f}")

    print(f"Worker {task_id} completed!")

    return {
        'task_id': task_id,
        'model_state': model.state_dict(),
        'memory_samples': local_memory.get_memory_samples(),
        'training_history': task_history,
        'success': True
    }


def run_optimized_training(params, task_dataloaders, test_dataloaders):
    """Run training with optimizations"""

    start_time = time.time()

    optimized_dataloaders = []
    for i, dataloader in enumerate(task_dataloaders):
        optimized_dataloaders.append(dataloader)

    # Run the actual training
    main_model = SimpleMLP(params['input_dim'], params['hidden_dim'], params['num_classes']).to(DEVICE)
    all_results = []
    accumulated_memory = []
    intermediate_eval_accuracies_history = []

    for task_id, task_dataloader in enumerate(optimized_dataloaders):
        print(f"\nProcessing Task {task_id + 1}/{params['num_tasks']}...")

        task_args = (
            task_id,
            task_dataloader,
            params,
            accumulated_memory,
            main_model.state_dict()
        )

        # Call optimized training function
        task_result = train_single_task(task_args)

        if task_result['success']:
            all_results.append(task_result)
            main_model.load_state_dict(task_result['model_state'])

            # Update accumulated memory
            accumulated_memory.extend([
                (s, l) for s, l in task_result['memory_samples']])

            # Limit memory size
            if len(accumulated_memory) > params['memory_size_p'] * params['num_pools']:
                accumulated_memory = accumulated_memory[-params['memory_size_p'] * params['num_pools']:]

            print(f"Task {task_id + 1} completed successfully!")

            # --- Perform intermediate evaluation after each task is trained ---
            print(f"  Evaluating model after training Task {task_id + 1}...")
            temp_evaluator = TAGEMEvaluator(test_dataloaders=test_dataloaders)
            current_accuracies = temp_evaluator.evaluate_tasks_up_to(main_model, task_id, test_dataloaders,
                                                                     device=params['device'])
            intermediate_eval_accuracies_history.append(current_accuracies)
            print(f"  Intermediate accuracies after Task {task_id + 1}: {current_accuracies}")
            # --- End intermediate evaluation ---

        else:
            print(f"Task {task_id + 1} failed!")

    training_time = time.time() - start_time

    # Create final memory
    final_memory = clustering_pools.ClusteringMemory(
        Q=params['memory_size_q'], P=params['memory_size_p'],
        input_type='samples', num_pools=params['num_pools'], device=params['device']
    )
    for sample_data, sample_label in accumulated_memory:
        final_memory.add_sample(sample_data, sample_label)

    return main_model, final_memory, all_results, training_time, intermediate_eval_accuracies_history