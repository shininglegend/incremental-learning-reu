import torch
import torch.nn as nn
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import time
import copy
from collections import defaultdict
import threading
from queue import Queue, Empty
import pickle
import os
import torch.optim as optim
import agem
import clustering
from simple_mlp import SimpleMLP
import config

params = config.get_params()
DEVICE = params['device']


class GradientAggregator:
    """Handles gradient aggregation from multiple parallel processes"""

    def __init__(self, model_template):
        self.model_template = model_template
        self.gradient_buffer = []
        self.lock = threading.Lock()

    def add_gradients(self, gradients, weight=1.0):
        """Add gradients from a worker with optional weighting"""
        with self.lock:
            self.gradient_buffer.append({
                'gradients': gradients,
                'weight': weight
            })

    def aggregate_and_apply(self, target_model, clear_buffer=True):
        """Aggregate all buffered gradients and apply to target model"""
        if not self.gradient_buffer:
            return

        with self.lock:
            # Initialize aggregated gradients
            aggregated_grads = {}
            total_weight = 0

            for grad_info in self.gradient_buffer:
                weight = grad_info['weight']
                gradients = grad_info['gradients']
                total_weight += weight

                for name, grad in gradients.items():
                    if name not in aggregated_grads:
                        aggregated_grads[name] = torch.zeros_like(grad)
                    aggregated_grads[name] += grad * weight

            # Average by total weight
            if total_weight > 0:
                for name in aggregated_grads:
                    aggregated_grads[name] /= total_weight

            # Apply to target model
            for name, param in target_model.named_parameters():
                if name in aggregated_grads and param.grad is not None:
                    param.grad.data = aggregated_grads[name]

            if clear_buffer:
                self.gradient_buffer.clear()


class PipelinedTrainingManager:
    """Implements pipelined training to overlap computation phases"""

    def __init__(self, params, num_pipeline_stages=3):
        self.params = params
        self.num_stages = num_pipeline_stages
        self.stage_queues = [Queue() for _ in range(num_pipeline_stages)]
        self.workers = []
        self.running = False

    def start_pipeline(self, model, task_dataloaders):
        """Start the training pipeline"""
        self.running = True

        # Stage 1: Data preprocessing and batching
        worker1 = threading.Thread(
            target=self._data_preprocessing_stage,
            args=(task_dataloaders,),
            daemon=True
        )

        # Stage 2: Forward pass and loss computation
        worker2 = threading.Thread(
            target=self._forward_pass_stage,
            args=(model,),
            daemon=True
        )

        # Stage 3: Backward pass and optimization
        worker3 = threading.Thread(
            target=self._optimization_stage,
            args=(model,),
            daemon=True
        )

        self.workers = [worker1, worker2, worker3]
        for worker in self.workers:
            worker.start()

    def stop_pipeline(self):
        """Stop the training pipeline"""
        self.running = False

        # Signal all queues to stop
        for queue in self.stage_queues:
            queue.put(None)  # Sentinel value

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)

    def _data_preprocessing_stage(self, task_dataloaders):
        """Stage 1: Preprocess and batch data"""
        for task_id, dataloader in enumerate(task_dataloaders):
            for batch_idx, (data, labels) in enumerate(dataloader):
                if not self.running:
                    break

                # Preprocess data (flatten, normalize, etc.)
                processed_data = {
                    'data': data.view(data.size(0), -1),
                    'labels': labels,
                    'task_id': task_id,
                    'batch_idx': batch_idx
                }

                self.stage_queues[0].put(processed_data)

        self.stage_queues[0].put(None)  # Signal end of data

    def _forward_pass_stage(self, model):
        """Stage 2: Forward pass and loss computation"""
        while self.running:
            try:
                batch_data = self.stage_queues[0].get(timeout=1.0)
                if batch_data is None:
                    break

                # Forward pass
                with torch.no_grad():
                    outputs = model(batch_data['data'])
                    loss = nn.CrossEntropyLoss()(outputs, batch_data['labels'])

                forward_result = {
                    **batch_data,
                    'outputs': outputs,
                    'loss': loss
                }

                self.stage_queues[1].put(forward_result)

            except Empty:
                continue

        self.stage_queues[1].put(None)  # Signal end

    def _optimization_stage(self, model):
        """Stage 3: Backward pass and optimization"""
        optimizer = optim.SGD(model.parameters(), lr=self.params['learning_rate'])

        while self.running:
            try:
                forward_result = self.stage_queues[1].get(timeout=1.0)
                if forward_result is None:
                    break

                # Backward pass
                optimizer.zero_grad()
                forward_result['loss'].backward()
                optimizer.step()

                # Optional: queue results for further processing
                self.stage_queues[2].put({
                    'task_id': forward_result['task_id'],
                    'batch_idx': forward_result['batch_idx'],
                    'loss': forward_result['loss'].item()
                })

            except Empty:
                continue


class GradientAccumulator:  # A simple helper for within-task accumulation
    def __init__(self, model):
        self.gradients = {name: torch.zeros_like(param).to(DEVICE) for name, param in model.named_parameters()}
        self.count = 0
        self.lock = threading.Lock()

    def add_gradients(self, grads_dict):
        with self.lock:
            for name, grad in grads_dict.items():
                if name in self.gradients:
                    self.gradients[name] += grad.to(DEVICE)  # Ensure accumulation on device
            self.count += 1

    def get_averaged_gradients(self):
        with self.lock:
            if self.count == 0:
                return None
            averaged = {name: grad / self.count for name, grad in self.gradients.items()}
            return averaged

    def reset(self):
        with self.lock:
            for name in self.gradients:
                self.gradients[name].zero_()
            self.count = 0

class AsyncModelUpdater:
    """ Handles asynchronous updates to the model parameters. Updates are queued and applied in a separate thread. """
    def __init__(self, model, update_frequency=5):
        self.model = model
        self.update_queue = Queue()
        self.update_frequency = update_frequency
        self.running = False
        self.update_thread = None
        self.lock = threading.Lock() # Lock for model parameter access

    def _update_loop(self):
        while self.running:
            try:
                updates = self.update_queue.get(timeout=1.0)
                if updates is None: # Sentinel for stopping
                    break
                with self.lock:
                    for name, update_value in updates.items():
                        if name in self.model.state_dict():
                            # Direct modification of parameter data
                            param = self.model.state_dict()[name]
                            # Ensure update_value is on the same device as the parameter
                            param.data.add_(update_value.to(param.device))
                self.update_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                print(f"Error in AsyncModelUpdater update loop: {e}")
                import traceback
                traceback.print_exc()

    def start(self):
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
            print("AsyncModelUpdater started.")

    def stop(self):
        if self.running:
            self.running = False
            self.update_queue.put(None) # Signal to stop
            self.update_thread.join(timeout=5.0)
            print("AsyncModelUpdater stopped.")

    def queue_update(self, updates):
        self.update_queue.put(updates)

class DistributedMemoryManager:
    """Manages a distributed memory for A-GEM across multiple shards/pools."""
    def __init__(self, params, num_shards=4):
        self.params = params
        self.num_shards = num_shards
        self.memory_shards = [clustering.ClusteringMemory(
            Q=params['memory_size_q'] // num_shards, # Distribute Q across shards
            P=params['memory_size_p'],
            input_type='samples',
            num_pools=params['num_pools'] // num_shards, # Distribute pools
            device='cpu' # Memory samples generally stored on CPU to reduce GPU memory pressure
        ) for _ in range(num_shards)]
        self.shard_locks = [threading.Lock() for _ in range(num_shards)]

    def add_sample(self, data, label):
        # Determine which shard to add to (e.g., round-robin or hash-based)
        shard_idx = hash(data.sum()) % self.num_shards # Simple hash-based distribution
        with self.shard_locks[shard_idx]:
            self.memory_shards[shard_idx].add_sample(data.to('cpu'), label.to('cpu')) # Ensure CPU

    def get_memory_samples(self):
        all_samples = []
        for i, shard in enumerate(self.memory_shards):
            with self.shard_locks[i]: # Acquire lock for each shard
                shard_samples = shard.get_memory_samples()
                # Limit samples per shard to avoid memory explosion
                if len(shard_samples) > (self.params['memory_size_p'] * self.params['memory_size_q'] / self.num_shards):
                    # Max samples per shard based on Q*P divided by num_shards
                    indices = np.random.choice(
                        len(shard_samples),
                        int(self.params['memory_size_p'] * self.params['memory_size_q'] / self.num_shards), # Adjust based on total desired samples
                        replace=False
                    )
                    shard_samples = [shard_samples[idx] for idx in indices]
                all_samples.extend(shard_samples)
        return all_samples

def train_task_with_advanced_parallelization(task_id, task_dataloader, params, shared_memory, main_model_state, MLP_class, device_to_use):
    """Train a single task using advanced parallelization strategies"""

    # Create model for this task and load state
    model = MLP_class(params['input_dim'], params['hidden_dim'], params['num_classes']).to(device_to_use)
    if main_model_state:
        model.load_state_dict(main_model_state)

    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # Create distributed memory manager
    memory_manager = DistributedMemoryManager(params, num_shards=4)

    # Add shared memory samples
    if shared_memory:
        for sample_data, sample_label in shared_memory:
            memory_manager.add_sample(sample_data.to('cpu'), sample_label.to('cpu'))  # Ensure memory samples are on CPU

    agem_handler = agem.AGEMHandler(model, criterion, optimizer)

    # For accumulating gradients within this task/process
    task_gradient_accumulator = GradientAggregator(model)  # Reusing GradientAggregator from this file

    try:
        task_history = []

        for epoch in range(params['num_epochs']):
            model.train()  # Make sure model is in train mode
            epoch_losses = []

            # Process batches with parallel workers
            with ThreadPoolExecutor(max_workers=4) as executor:
                batch_futures = []

                for batch_idx, (data, labels) in enumerate(task_dataloader):
                    # Move data to device
                    data = data.to(device_to_use)
                    labels = labels.to(device_to_use)

                    # Submit batch for parallel processing
                    future = executor.submit(
                        _process_batch_hybrid_grad_return,  # New function to return gradients
                        data, labels, model.state_dict(),  # Pass model state to worker function
                        memory_manager, params, MLP_class, device_to_use
                    )
                    batch_futures.append(future)

                # Collect results
                for future in as_completed(batch_futures):
                    result = future.result()
                    if result['success']:
                        if result['loss'] is not None:
                            epoch_losses.append(result['loss'])

                        for i in range(len(result['data'])):
                            memory_manager.add_sample(result['data'][i].to('cpu'), result['labels'][i].to('cpu'))

            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            task_history.append(avg_loss)

            if epoch % 5 == 0:
                print(f"Task {task_id}, Epoch {epoch + 1}/{params['num_epochs']}: Loss = {avg_loss:.4f}")

        # At the end of the task (for this worker/process), return its final model state
        return {
            'task_id': task_id,
            'model_state': model.state_dict(),  # Return the worker's final model state
            'memory_samples': memory_manager.get_memory_samples(),
            'training_history': task_history,
            'success': True
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'task_id': task_id,
            'model_state': model.state_dict(),
            'memory_samples': memory_manager.get_memory_samples(),
            'training_history': [],
            'success': False,
            'error': str(e)
        }


# New _process_batch_hybrid_grad_return to compute and return gradients for a single batch
# This function is run by threads and needs a separate model instance/state
def _process_batch_hybrid_grad_return(data, labels, model_state, memory_manager, params, MLP_class, device_to_use):
    """
    Process a single batch in hybrid training mode, returning gradients.
    This function creates a temporary model for gradient computation.
    """
    # Create a temporary model instance for this batch calculation
    temp_model = MLP_class(params['input_dim'], params['hidden_dim'], params['num_classes']).to(device_to_use)
    temp_model.load_state_dict(model_state)  # Load the current state passed from the main worker
    temp_optimizer = torch.optim.SGD(temp_model.parameters(), lr=params['learning_rate'])
    temp_criterion = nn.CrossEntropyLoss()
    temp_agem_handler = agem.AGEMHandler(temp_model, temp_criterion, temp_optimizer)

    try:
        # Get memory samples for gradient projection
        # Ensure memory samples are on the correct device for the temporary model
        memory_samples = [(s.to(device_to_use), l.to(device_to_use)) for s, l in memory_manager.get_memory_samples()]

        # Perform A-GEM optimization on the temporary model
        batch_loss = temp_agem_handler.optimize(data, labels, memory_samples)

        return {
            'loss': batch_loss,
            'samples_processed': len(data),
            'data': data.to('cpu'),  # For memory
            'labels': labels.to('cpu'),  # For memory
            'success': True
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'loss': None,
            'samples_processed': 0,
            'error': str(e),
            'data': data.to('cpu'),
            'labels': labels.to('cpu'),
            'success': False
        }


def _run_hybrid_training(params, model, task_dataloaders):
    """Hybrid approach combining multiple optimization strategies"""
    print("Starting Hybrid Parallel training...")
    start_time = time.time()

    gradient_aggregator = GradientAggregator(model) # Model instance
    async_updater = AsyncModelUpdater(model, update_frequency=5)  # Now defined
    distributed_memory = DistributedMemoryManager(params, num_shards=4)

    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    agem_handler = agem.AGEMHandler(model, criterion, optimizer)

    async_updater.start()

    try:
        total_samples = 0
        training_history = []
        all_results = [] # To store results for eventual merging or analysis

        # Process each task sequentially, but with hybrid batch processing
        for task_id, task_dataloader in enumerate(task_dataloaders):
            print(f"\n--- Training on Task {task_id} ---")
            task_start_time = time.time()

            # Pass current model state and memory to parallel workers
            current_model_state = model.state_dict()
            current_shared_memory = distributed_memory.get_memory_samples()

            # Use ProcessPoolExecutor to run train_task_with_advanced_parallelization for each task
            # (simulating hybrid, but still within a single process here by calling directly for simplicity)

            task_result = train_task_with_advanced_parallelization(
                task_id, task_dataloader, params,
                current_shared_memory, current_model_state,
                SimpleMLP, DEVICE
            )

            if task_result['success']:
                all_results.append(task_result)
                # Update the main model with the state from the completed task's worker
                model.load_state_dict(task_result['model_state'])
                # Update the main distributed memory manager with samples from the worker
                # Clear existing memory and rebuild from worker's samples to avoid duplicates
                # or merge strategically if needed. For simplicity, just adding them.
                for sample_data, sample_label in task_result['memory_samples']:
                    distributed_memory.add_sample(sample_data.to('cpu'), sample_label.to('cpu')) # Ensure CPU

            else:
                print(f"Task {task_id} failed with error: {task_result.get('error', 'Unknown error')}")

            task_time = time.time() - task_start_time
            print(f"Task {task_id} completed in {task_time:.2f}s")


    finally:
        async_updater.stop()

    training_time = time.time() - start_time
    return model, distributed_memory, all_results, training_time


def _process_batch_hybrid(data, labels, model, memory_manager, agem_handler, params):
    """Process a single batch in hybrid training mode"""
    try:
        # Get memory samples for gradient projection
        memory_samples = memory_manager.get_memory_samples()
        # Ensure memory samples are on the device of the model.
        memory_samples_on_device = [(s.to(DEVICE), l.to(DEVICE)) for s, l in memory_samples]

        # Perform optimization with A-GEM
        batch_loss = agem_handler.optimize(data, labels, memory_samples_on_device)

        # Add samples to distributed memory
        for i in range(len(data)):
            memory_manager.add_sample(data[i].to('cpu'), labels[i].to('cpu'))

        # Compute parameter updates (simplified)
        # This part might not yield meaningful updates if agem_handler.optimize
        # already performs optimizer.step() and clears gradients.
        parameter_updates = {}
        if batch_loss is not None:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Cloning and scaling gradients for an "update"
                    # The value 0.01 is arbitrary and might need tuning based on how updates are applied.
                    parameter_updates[name] = param.grad.clone() * 0.01

        return {
            'loss': batch_loss,
            'samples_processed': len(data),
            'parameter_updates': parameter_updates
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'loss': None,
            'samples_processed': 0,
            'error': str(e),
            'parameter_updates': None
        }

def _evaluate_model_quick(model, dataloader):
    """Quick model evaluation"""
    correct = 0
    total = 0
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        for data, labels in dataloader:
            if len(data.shape) > 2:
                data = data.view(data.size(0), -1)
            # Ensure data is on the correct device for evaluation
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

class ModelEnsemble:
    """Manages an ensemble of models for parallel training and merging."""
    def __init__(self, model_template, num_models=4):
        self.model_template = model_template
        self.num_models = num_models
        self.models = [model_template(784, 200, 10).to(DEVICE) for _ in range(num_models)] # Example init

    def train_ensemble_parallel(self, task_dataloaders, params):
        """Train each model in the ensemble on a subset of data or tasks in parallel."""
        # For simplicity, each model gets a copy of all task_dataloaders
        # In a more complex scenario, data partitioning would be needed.
        training_args = []
        for i in range(self.num_models):
            # Each ensemble member trains independently
            # Pass model state and data. Data should ideally be partitioned.
            # For this example, we'll just pass all dataloaders to each,
            # assuming it's a simplification for a full ensemble method.
            training_args.append({
                'model_id': i,
                'model_state': self.models[i].state_dict(),
                'params': params,
                'task_dataloaders': task_dataloaders # All dataloaders for each ensemble member
            })

        results = []
        # Use ProcessPoolExecutor to train ensemble members in parallel
        with ProcessPoolExecutor(max_workers=self.num_models) as executor:
            futures = [executor.submit(self._train_ensemble_member, arg) for arg in training_args]
            for future in as_completed(futures):
                results.append(future.result())
        return results

    def _train_ensemble_member(self, args):
        """Internal function to train a single ensemble member."""
        model_id = args['model_id']
        model_state = args['model_state']
        params = args['params']
        task_dataloaders = args['task_dataloaders']

        model = self.model_template(params['input_dim'], params['hidden_dim'], params['num_classes']).to(DEVICE)
        model.load_state_dict(model_state)
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        agem_handler = agem.AGEMHandler(model, criterion, optimizer)
        memory = clustering.ClusteringMemory(Q=params['memory_size_q'], P=params['memory_size_p'],
                                            input_type='samples', num_pools=params['num_pools'], device=params['device'])

        print(f"Ensemble Member {model_id} starting training...")
        for task_id, task_dataloader in enumerate(task_dataloaders):
            for epoch in range(params['num_epochs']):
                model.train()
                epoch_losses = []
                for batch_idx, (data, labels) in enumerate(task_dataloader):
                    data = data.to(DEVICE)
                    labels = labels.to(DEVICE)
                    batch_loss = agem_handler.optimize(data, labels, memory.get_memory_samples())
                    if batch_loss is not None:
                        epoch_losses.append(batch_loss)
                    for i in range(len(data)):
                        memory.add_sample(data[i], labels[i])
                # print(f"  Ensemble Member {model_id}, Task {task_id}, Epoch {epoch}: Loss = {np.mean(epoch_losses):.4f}")
        print(f"Ensemble Member {model_id} completed training.")
        return {'model_id': model_id, 'model_state': model.state_dict(), 'success': True}

    def merge_ensemble(self, training_results):
        """Merge the trained models, e.g., by averaging their weights."""
        if not training_results:
            return None

        # Average model parameters
        merged_state = {}
        first_state = training_results[0]['model_state']

        for key in first_state.keys():
            merged_state[key] = torch.zeros_like(first_state[key])

        for result in training_results:
            model_state = result['model_state']
            for key in model_state.keys():
                merged_state[key] += model_state[key]

        # Average
        for key in merged_state.keys():
            merged_state[key] /= len(training_results)

        # Load into first model
        self.models[0].load_state_dict(merged_state)
        return self.models[0]


def run_ensemble_parallel_training(params, model_template, task_dataloaders):
    """Run training with model ensemble parallelization"""
    print("Starting ensemble parallel training...")

    start_time = time.time()

    # Create ensemble
    ensemble = ModelEnsemble(model_template, num_models=4)

    # Train ensemble in parallel
    training_results = ensemble.train_ensemble_parallel(task_dataloaders, params)

    # Merge models
    merged_model = ensemble.merge_ensemble(training_results)

    training_time = time.time() - start_time

    # Create dummy memory for compatibility
    dummy_memory = clustering.ClusteringMemory(
        Q=params['memory_size_q'], P=params['memory_size_p'],
        input_type='samples', num_pools=params['num_pools']
    )

    return merged_model, dummy_memory, training_results, training_time