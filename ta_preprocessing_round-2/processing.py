import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

import clustering
import agem
from simple_mlp import SimpleMLP
from load_dataset import optimize_dataloader

# config
from config import params
DEVICE = params['device']


def run_sequential_training(params, task_dataloaders_nested):
    """Baseline sequential training"""
    print("Running sequential training...")

    model = SimpleMLP(params['input_dim'], params['hidden_dim'], params['num_classes'])
    model.to(params['device'])
    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    memory = clustering.ClusteringMemory(
        Q=params['memory_size_q'], P=params['memory_size_p'],
        input_type='samples', num_pools=params['num_pools'], device=params['device']
    )
    agem_handler = agem.AGEMHandler(model, criterion, optimizer, device=params['device'])

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

                batch_loss = agem_handler.optimize(data, labels, memory.get_memory_samples())

                if batch_loss is not None:
                    epoch_losses.append(batch_loss)

                for i in range(len(data)):
                    memory.add_sample(data[i], labels[i])

    training_time = time.time() - start_time
    return model, memory, training_time


def train_single_task_parallel(args):
    """Train a single task with reliable optimizations (no threading)"""
    task_id, task_dataloader, params, shared_memory_samples, model_state_dict = args

    print(f"Worker starting Task {task_id}")

    model = SimpleMLP(params['input_dim'], params['hidden_dim'], params['num_classes']).to(DEVICE)
    if model_state_dict:
        model.load_state_dict(model_state_dict)

    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    local_memory = clustering.ClusteringMemory(
        Q=params['memory_size_q'],
        P=params['memory_size_p'],
        input_type='samples',
        num_pools=params['num_pools'],
        device=params['device']
    )

    if shared_memory_samples:
        for sample_data, sample_label in shared_memory_samples:
            local_memory.add_sample(sample_data.to('cpu'), sample_label.to('cpu'))

    agem_handler = agem.AGEMHandler(model, criterion, optimizer, device=params['device'])
    task_history = []

    # OPTIMIZATION 1: Larger effective batch size through gradient accumulation
    accumulation_steps = 4  # Effectively 4x larger batches

    # OPTIMIZATION 2: Mixed precision training (if available)
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

            if use_amp:
                with torch.amp.autocast('cuda'):
                    memory_samples = local_memory.get_memory_samples()
                    memory_samples_on_device = [
                        (s.to(DEVICE), l.to(DEVICE)) for s, l in memory_samples
                    ]

                    projected_grads_flat, batch_loss = agem_handler.compute_and_project_batch_gradient(
                        data, labels, memory_samples_on_device
                    )
            else:
                memory_samples = local_memory.get_memory_samples()
                memory_samples_on_device = [
                    (s.to(DEVICE), l.to(DEVICE)) for s, l in memory_samples
                ]

                projected_grads_flat, batch_loss = agem_handler.compute_and_project_batch_gradient(
                    data, labels, memory_samples_on_device
                )

            if batch_loss is not None:
                epoch_losses.append(batch_loss)

            if projected_grads_flat is not None:
                if accumulated_projected_grads is None:
                    accumulated_projected_grads = torch.zeros_like(projected_grads_flat).to(DEVICE)

                accumulated_projected_grads += projected_grads_flat
                step_count += 1

                # Apply accumulated gradients every N steps
                if step_count % accumulation_steps == 0:
                    agem_handler.apply_accumulated_gradients(accumulated_projected_grads)
                    accumulated_projected_grads = None  # Reset accumulation
                    step_count = 0

            # Add samples to memory
            for i in range(len(data)):
                local_memory.add_sample(data[i].to('cpu'), labels[i].to('cpu'))

        # Apply any remaining accumulated gradients
        if accumulated_projected_grads is not None and step_count > 0:
            agem_handler.apply_accumulated_gradients(accumulated_projected_grads)

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


def process_batch_parallel(agem_handler, data, labels, memory):
    """Process a single batch in parallel - thread-safe version"""
    try:
        # Create a copy of agem_handler for thread safety
        # This is crucial - sharing the same handler across threads causes issues
        import copy
        local_agem_handler = copy.deepcopy(agem_handler)

        # Move data to correct device
        data = data.to(local_agem_handler.device)
        labels = labels.to(local_agem_handler.device)

        # Get memory samples (thread-safe access)
        memory_samples = memory.get_memory_samples()
        memory_samples_on_device = [
            (s.to(local_agem_handler.device), l.to(local_agem_handler.device))
            for s, l in memory_samples
        ]

        # Compute projected gradients and batch loss
        projected_grads_flat, batch_loss = local_agem_handler.compute_and_project_batch_gradient(
            data, labels, memory_samples_on_device
        )

        return {
            'loss': batch_loss,
            'projected_gradients_flat': projected_grads_flat,
            'data': data.to('cpu'),
            'labels': labels.to('cpu'),
            'success': True
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'loss': None,
            'projected_gradients_flat': None,
            'data': data.to('cpu') if 'data' in locals() else None,
            'labels': labels.to('cpu') if 'labels' in locals() else None,
            'success': False,
            'error': str(e)
        }


def run_optimized_training(params, task_dataloaders):
    """Run training with optimizations"""

    start_time = time.time()

    optimized_dataloaders = []
    for i, dataloader in enumerate(task_dataloaders):
        optimized_dataloaders.append(dataloader)

    # Run the actual training
    main_model = SimpleMLP(params['input_dim'], params['hidden_dim'], params['num_classes']).to(DEVICE)
    all_results = []
    accumulated_memory = []

    for task_id, task_dataloader in enumerate(optimized_dataloaders):
        print(f"\nProcessing Task {task_id + 1}/{params['num_tasks']}...")

        task_args = (
            task_id,
            task_dataloader,  # Just pass the original DataLoader
            params,
            accumulated_memory,
            main_model.state_dict()
        )

        # Call optimized training function
        task_result = train_single_task_parallel(task_args)

        if task_result['success']:
            all_results.append(task_result)
            main_model.load_state_dict(task_result['model_state'])

            # Update accumulated memory
            accumulated_memory.extend([
                (s.to('cpu'), l.to('cpu')) for s, l in task_result['memory_samples']
            ])

            # Limit memory size
            if len(accumulated_memory) > params['memory_size_p'] * params['num_pools']:
                accumulated_memory = accumulated_memory[-params['memory_size_p'] * params['num_pools']:]

            print(f"Task {task_id + 1} completed successfully!")
        else:
            print(f"Task {task_id + 1} failed!")

    training_time = time.time() - start_time

    # Create final memory
    final_memory = clustering.ClusteringMemory(
        Q=params['memory_size_q'], P=params['memory_size_p'],
        input_type='samples', num_pools=params['num_pools']
    )
    for sample_data, sample_label in accumulated_memory:
        final_memory.add_sample(sample_data, sample_label)

    return main_model, final_memory, all_results, training_time


def merge_task_models(task_results, params):
    """Merge models from different tasks"""
    if not task_results:
        return None

    # Create base model
    merged_model = SimpleMLP(params['input_dim'], params['hidden_dim'], params['num_classes'])

    # Use the last task's model as the base (most recent learning)
    if task_results:
        last_task_state = task_results[-1]['model_state']
        merged_model.load_state_dict(last_task_state)

    return merged_model