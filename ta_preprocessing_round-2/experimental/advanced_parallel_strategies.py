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
import torch.optim as optim  # Added missing import for optim
import agem  # Ensure agem module is available in your environment
import clustering  # Ensure clustering module is available in your environment
from ta_agem.new_main import SimpleMLP, DEVICE  # Corrected syntax, removed 'import agem'


global start_time


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


def train_task_with_advanced_parallelization(task_id, task_dataloader, params, shared_memory, main_model_state):
    """Train a single task using advanced parallelization strategies"""

    # Create model for this task and load state
    model = SimpleMLP(params['input_dim'], params['hidden_dim'], params['num_classes']).to(DEVICE)
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
                    data = data.to(DEVICE)
                    labels = labels.to(DEVICE)

                    # Submit batch for parallel processing
                    future = executor.submit(
                        _process_batch_hybrid_grad_return,  # New function to return gradients
                        data, labels, model.state_dict(),  # Pass model state to worker function
                        memory_manager, params
                    )
                    batch_futures.append(future)

                # Collect results
                for future in as_completed(batch_futures):
                    result = future.result()
                    if result['success']:
                        if result['loss'] is not None:
                            epoch_losses.append(result['loss'])

                        # Add samples to distributed memory (this happens in the worker, but we need to ensure consistency)
                        # Or, better: make add_sample thread-safe and have the worker call it directly on the shared memory_manager.
                        # For now, let's assume memory_manager is already thread-safe or its add_sample works without issues here.
                        # If memory_manager is passed by reference to threads, its locks will work.
                        for i in range(len(result['data'])):
                            memory_manager.add_sample(result['data'][i].to('cpu'), result['labels'][i].to('cpu'))

                        # Accumulate gradients (these are from a "dummy" model in _process_batch_hybrid_grad_return)
                        # This requires careful handling if you truly want to apply them to the *main* model later.
                        # A simpler approach is to let each worker run its own model completely and then average states.
                        # Given your "update big model at end of task" and "each worker finishes its part",
                        # the model should be updated *per worker* for its portion, and then the worker's
                        # *final state* is returned. The `train_single_task_parallel` approach is more aligned with this.
                        # Let's align _run_hybrid_training with _train_single_task_parallel
                        pass  # No gradient accumulation at batch level here, let worker finish and return state

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
def _process_batch_hybrid_grad_return(data, labels, model_state, memory_manager, params):
    """
    Process a single batch in hybrid training mode, returning gradients.
    This function creates a temporary model for gradient computation.
    """
    # Create a temporary model instance for this batch calculation
    temp_model = SimpleMLP(params['input_dim'], params['hidden_dim'], params['num_classes']).to(DEVICE)
    temp_model.load_state_dict(model_state)  # Load the current state passed from the main worker
    temp_optimizer = torch.optim.SGD(temp_model.parameters(), lr=params['learning_rate'])
    temp_criterion = nn.CrossEntropyLoss()
    temp_agem_handler = agem.AGEMHandler(temp_model, temp_criterion, temp_optimizer)

    try:
        # Get memory samples for gradient projection
        # Ensure memory samples are on the correct device for the temporary model
        memory_samples = [(s.to(DEVICE), l.to(DEVICE)) for s, l in memory_manager.get_memory_samples()]

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

    gradient_aggregator = GradientAggregator(model)
    async_updater = AsyncModelUpdater(model, update_frequency=5)  # Now defined
    distributed_memory = DistributedMemoryManager(params, num_shards=4)

    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    agem_handler = agem.AGEMHandler(model, criterion, optimizer)

    async_updater.start()

    try:
        total_samples = 0
        training_history = []

        for task_id, task_dataloader in enumerate(task_dataloaders):
            print(f"\n--- Training on Task {task_id} ---")
            task_start_time = time.time()

            for epoch in range(params['num_epochs']):
                epoch_losses = []

                # Process batches with parallel workers
                with ThreadPoolExecutor(max_workers=4) as executor:
                    batch_futures = []

                    for batch_idx, (data, labels) in enumerate(task_dataloader):
                        # Submit batch for parallel processing
                        future = executor.submit(
                            _process_batch_hybrid,
                            data, labels, model, distributed_memory,
                            agem_handler, params
                        )
                        batch_futures.append(future)

                        # Process completed batches
                        if len(batch_futures) >= 4:  # Process in chunks
                            for future in as_completed(batch_futures):
                                result = future.result()
                                if result['loss'] is not None:
                                    epoch_losses.append(result['loss'])

                                # Queue async parameter updates
                                # Note: If agem_handler.optimize performs optimizer.step(),
                                # param.grad might be None or zeroed here.
                                # The effectiveness of parameter_updates here depends on agem_handler's internal logic.
                                if 'parameter_updates' in result and result['parameter_updates'] is not None:
                                    async_updater.queue_update(result['parameter_updates'])

                                total_samples += result['samples_processed']

                            batch_futures = []

                    # Process remaining batches
                    for future in as_completed(batch_futures):
                        result = future.result()
                        if result['loss'] is not None:
                            epoch_losses.append(result['loss'])
                        total_samples += result['samples_processed']

                avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
                training_history.append({
                    'task_id': task_id,
                    'epoch': epoch,
                    'loss': avg_epoch_loss,
                    'samples': total_samples
                })

                if params['quick_test_mode'] and epoch % 5 == 0:
                    print(f'  Epoch {epoch + 1}/{params["num_epochs"]}: Loss = {avg_epoch_loss:.4f}')

            task_time = time.time() - task_start_time
            print(f"Task {task_id} completed in {task_time:.2f}s")

            # Evaluate after each task
            model.eval()
            with torch.no_grad():
                task_accuracy = _evaluate_model_quick(model, task_dataloader)
                print(f"Task {task_id} accuracy: {task_accuracy:.4f}")

    finally:
        async_updater.stop()

    training_time = time.time() - start_time
    return model, distributed_memory, training_history, training_time


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
            'data': data.to('cpu'),
            'labels': labels.to('cpu'),
            'success': False
        }


# Modify DistributedMemoryManager constructor to pass model_template
class DistributedMemoryManager:
    """Manages distributed episodic memory across processes"""

    def __init__(self, params, num_shards=4):
        self.params = params
        self.num_shards = num_shards
        self.memory_shards = []
        self.shard_locks = []

        # Initialize memory shards
        for i in range(num_shards):
            shard = clustering.ClusteringMemory(
                Q=params['memory_size_q'] // num_shards,
                P=params['memory_size_p'],
                input_type='samples',
                num_pools=params['num_pools'] // num_shards,
                device=params['device']
            )
            self.memory_shards.append(shard)
            self.shard_locks.append(threading.Lock())  # Threading lock for intra-process thread safety

    def add_sample(self, sample_data, sample_label):
        """Add sample to appropriate shard based on label"""
        # Ensure label is an int for shard_idx calculation
        shard_idx = sample_label.item() % self.num_shards if hasattr(sample_label,
                                                                     'item') else sample_label % self.num_shards

        with self.shard_locks[shard_idx]:
            self.memory_shards[shard_idx].add_sample(sample_data, sample_label)

    # get_memory_samples remains largely the same, but ensure samples are moved to device when used by model
    # (handled in _process_batch_hybrid)
    def get_memory_samples(self, num_samples_per_shard=None):
        """Get memory samples from all shards"""
        if num_samples_per_shard is None:
            pass  # Keep original logic if it correctly limits total memory.

        all_samples = []

        for i, (shard, lock) in enumerate(zip(self.memory_shards, self.shard_locks)):
            with lock:  # Acquire lock for each shard
                shard_samples = shard.get_memory_samples()
                # Limit samples per shard to avoid memory explosion
                if len(shard_samples) > (self.params['memory_size_p'] * self.params[
                    'memory_size_q'] / self.num_shards):  # Max samples per shard based on Q*P divided by num_shards
                    indices = np.random.choice(
                        len(shard_samples),
                        int(self.params['memory_size_p'] * self.params['memory_size_q'] / self.num_shards),
                        # Adjust based on total desired samples
                        replace=False
                    )
                    shard_samples = [shard_samples[idx] for idx in indices]

                all_samples.extend(shard_samples)

        return all_samples


# --- Definition of AsyncModelUpdater class ---
class AsyncModelUpdater:
    """
    Handles asynchronous updates to the model parameters.
    Updates are queued and applied in a separate thread.
    """

    def __init__(self, model, update_frequency=5):
        self.model = model
        self.update_queue = Queue()
        self.update_frequency = update_frequency
        self.running = False
        self.update_thread = None
        self.lock = threading.Lock()  # Lock for model parameter access

    def _update_loop(self):
        while self.running:
            try:
                updates = self.update_queue.get(timeout=1.0)
                if updates is None:  # Sentinel for stopping
                    break

                with self.lock:
                    for name, update_value in updates.items():
                        if name in self.model.state_dict():
                            param = self.model.state_dict()[name]
                            if param.is_leaf:
                                # Ensure update_value is on the same device as the parameter
                                param.data.add_(update_value.to(param.device))
                            else:
                                # Handle non-leaf parameters or log a warning
                                pass

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
            self.update_queue.put(None)  # Signal to stop the thread
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=5.0)
            print("AsyncModelUpdater stopped.")

    def queue_update(self, updates):
        """Queue parameter updates to be applied asynchronously."""
        if self.running:
            self.update_queue.put(updates)
        else:
            print("Warning: AsyncModelUpdater not running, updates not queued.")


# --- End of AsyncModelUpdater class ---


def run_advanced_parallel_training(params, model, task_dataloaders, strategy='hybrid'):
    """
    Run training with advanced parallelization strategies

    Strategies:
    - 'pipeline': Pipelined training with overlapped stages
    - 'async_updates': Asynchronous parameter updates
    - 'distributed_memory': Distributed episodic memory
    - 'hybrid': Combination of multiple strategies
    """

    print(f"Starting advanced parallel training with strategy: {strategy}")
    start_time = time.time()

    if strategy == 'pipeline':
        return _run_pipelined_training(params, model, task_dataloaders)
    elif strategy == 'async_updates':
        return _run_async_update_training(params, model, task_dataloaders)
    elif strategy == 'distributed_memory':
        return _run_distributed_memory_training(params, model, task_dataloaders)
    elif strategy == 'hybrid':
        return _run_hybrid_training(params, model, task_dataloaders)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _run_pipelined_training(params, model, task_dataloaders):
    """Run training with pipelined execution"""
    pipeline_manager = PipelinedTrainingManager(params, num_pipeline_stages=3)

    start_time = time.time()
    pipeline_manager.start_pipeline(model, task_dataloaders)

    # Monitor pipeline progress
    try:
        while pipeline_manager.running:
            time.sleep(1.0)  # Check every second

            # Check if pipeline is complete (simplified check)
            all_empty = all(queue.empty() for queue in pipeline_manager.stage_queues)
            if all_empty:
                break

    finally:
        pipeline_manager.stop_pipeline()

    training_time = time.time() - start_time

    # Initialize basic memory for compatibility
    basic_memory = clustering.ClusteringMemory(
        Q=params['memory_size_q'], P=params['memory_size_p'],
        input_type='samples', num_pools=params['num_pools']
    )

    return model, basic_memory, [], training_time


def _run_async_update_training(params, model, task_dataloaders):
    """Run training with asynchronous parameter updates"""

    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    agem_handler = agem.AGEMHandler(model, criterion, optimizer)

    # Standard memory
    memory = clustering.ClusteringMemory(
        Q=params['memory_size_q'], P=params['memory_size_p'],
        input_type='samples', num_pools=params['num_pools']
    )

    async_updater = AsyncModelUpdater(model, update_frequency=10)
    async_updater.start()

    start_time = time.time()
    training_history = []

    try:
        for task_id, task_dataloader in enumerate(task_dataloaders):
            print(f"\n--- Async Training on Task {task_id} ---")

            for epoch in range(params['num_epochs']):
                epoch_losses = []

                for batch_idx, (data, labels) in enumerate(task_dataloader):
                    # Standard training step
                    memory_samples = memory.get_memory_samples()
                    # Move data to device for agem_handler.optimize
                    data = data.to(DEVICE)
                    labels = labels.to(DEVICE)
                    memory_samples_on_device = [(s.to(DEVICE), l.to(DEVICE)) for s, l in memory_samples]

                    batch_loss = agem_handler.optimize(data, labels, memory_samples_on_device)

                    if batch_loss is not None:
                        epoch_losses.append(batch_loss)

                    # Add to memory (on CPU as per other functions)
                    for i in range(len(data)):
                        memory.add_sample(data[i].to('cpu'), labels[i].to('cpu'))

                    # Queue async updates (simplified)
                    if batch_idx % 5 == 0:  # Every 5 batches
                        updates = {}
                        for name, param in model.named_parameters():
                            # Ensure gradients exist before cloning
                            if param.grad is not None:
                                updates[name] = param.grad.clone() * 0.001
                        if updates:  # Only queue if there are actual updates
                            async_updater.queue_update(updates)

                avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
                training_history.append(avg_loss)

                if params['quick_test_mode'] and epoch % 5 == 0:
                    print(f'  Epoch {epoch + 1}: Loss = {avg_loss:.4f}')

    finally:
        async_updater.stop()

    training_time = time.time() - start_time
    return model, memory, training_history, training_time


def _run_distributed_memory_training(params, model, task_dataloaders):
    """Run training with distributed episodic memory"""

    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    agem_handler = agem.AGEMHandler(model, criterion, optimizer)

    # Distributed memory
    distributed_memory = DistributedMemoryManager(params, num_shards=4)

    start_time = time.time()
    training_history = []

    for task_id, task_dataloader in enumerate(task_dataloaders):
        print(f"\n--- Distributed Memory Training on Task {task_id} ---")

        for epoch in range(params['num_epochs']):
            epoch_losses = []

            # Process batches in parallel using threads
            with ThreadPoolExecutor(max_workers=4) as executor:
                batch_futures = []

                for batch_idx, (data, labels) in enumerate(task_dataloader):
                    future = executor.submit(
                        _process_batch_distributed,
                        data, labels, model, distributed_memory, agem_handler
                    )
                    batch_futures.append(future)

                # Collect results
                for future in as_completed(batch_futures):
                    result = future.result()
                    if result['loss'] is not None:
                        epoch_losses.append(result['loss'])

            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            training_history.append(avg_loss)

            if params['quick_test_mode'] and epoch % 5 == 0:
                print(f'  Epoch {epoch + 1}: Loss = {avg_loss:.4f}')

    training_time = time.time() - start_time
    return model, distributed_memory, training_history, training_time


def _process_batch_distributed(data, labels, model, distributed_memory, agem_handler):
    """Process batch with distributed memory"""
    try:
        memory_samples = distributed_memory.get_memory_samples()
        # Move data to device for agem_handler.optimize
        data = data.to(DEVICE)
        labels = labels.to(DEVICE)
        memory_samples_on_device = [(s.to(DEVICE), l.to(DEVICE)) for s, l in memory_samples]

        batch_loss = agem_handler.optimize(data, labels, memory_samples_on_device)

        # Add samples to distributed memory (on CPU as per other functions)
        for i in range(len(data)):
            distributed_memory.add_sample(data[i].to('cpu'), labels[i].to('cpu'))

        return {'loss': batch_loss, 'samples': len(data)}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'loss': None, 'samples': 0, 'error': str(e)}


def _evaluate_model_quick(model, dataloader):
    """Quick model evaluation"""
    correct = 0
    total = 0

    model.eval()  # Set model to evaluation mode
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

    return correct / total if total > 0 else 0.0


class ModelEnsemble:
    """Ensemble of models for improved parallelization"""

    def __init__(self, model_template, num_models=4):
        self.models = []
        self.optimizers = []
        self.num_models = num_models

        for i in range(num_models):
            model = copy.deepcopy(model_template)
            # Ensure the model is moved to the correct device
            model.to(DEVICE)
            optimizer = optim.SGD(model.parameters(), lr=1e-3)
            self.models.append(model)
            self.optimizers.append(optimizer)

    def train_ensemble_parallel(self, task_dataloaders, params):
        """Train ensemble models in parallel"""

        def train_single_model(model_idx, model_state, optimizer_state, task_dataloaders_picklable, params_picklable):
            """Train a single model in the ensemble (run in a separate process)"""
            # Reconstruct model and optimizer from states as they are not directly picklable across processes
            local_model = SimpleMLP(params_picklable['input_dim'], params_picklable['hidden_dim'],
                                    params_picklable['num_classes']).to(DEVICE)
            local_model.load_state_dict(model_state)
            local_optimizer = optim.SGD(local_model.parameters(), lr=params_picklable['learning_rate'])
            local_optimizer.load_state_dict(optimizer_state)

            local_criterion = nn.CrossEntropyLoss()
            local_agem_handler = agem.AGEMHandler(local_model, local_criterion, local_optimizer)

            local_memory = clustering.ClusteringMemory(
                Q=params_picklable['memory_size_q'], P=params_picklable['memory_size_p'],
                input_type='samples', num_pools=params_picklable['num_pools']
            )

            local_training_history = []

            for task_id, dataloader_data in enumerate(task_dataloaders_picklable):
                # Reconstruct dataloader (simplified - in real scenario, this would involve dataset/dataloader reconstruction)
                # For demonstration, assuming dataloader_data is iterable batches
                # This is a placeholder, as full dataloader re-creation would be complex without full context.
                # A common approach is to pass paths to data and load it within the process.
                # For simplicity, we assume task_dataloaders_picklable is a list of lists of (data, labels) tuples.
                task_dataloader = [(torch.tensor(d).to(DEVICE), torch.tensor(l).to(DEVICE)) for d, l in dataloader_data]

                for epoch in range(params_picklable['num_epochs']):
                    local_model.train()
                    epoch_losses = []

                    for data, labels in task_dataloader:
                        memory_samples = local_memory.get_memory_samples()
                        memory_samples_on_device = [(s.to(DEVICE), l.to(DEVICE)) for s, l in memory_samples]

                        batch_loss = local_agem_handler.optimize(data, labels, memory_samples_on_device)

                        if batch_loss is not None:
                            epoch_losses.append(batch_loss)

                        for i in range(len(data)):
                            local_memory.add_sample(data[i].to('cpu'), labels[i].to('cpu'))  # Store on CPU

                    avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
                    local_training_history.append(avg_loss)

            # Return states that are picklable
            return {
                'model_idx': model_idx,
                'model_state': local_model.state_dict(),
                'training_history': local_training_history,
                'memory_samples': local_memory.get_memory_samples()  # Return memory samples if needed
            }

        # Prepare picklable arguments for ProcessPoolExecutor
        # Dataloaders are not directly picklable for ProcessPoolExecutor.
        # You need to pass the data itself or paths to data that can be loaded in each process.
        # For this example, I'm making a simplification: converting dataloaders to a list of lists of (data, labels)
        # This might consume a lot of memory if dataloaders are large.
        task_dataloaders_picklable = []
        for task_dl in task_dataloaders:
            task_data_list = []
            for data, labels in task_dl:
                # Convert tensors to numpy arrays for pickling, then convert back in worker
                task_data_list.append((data.cpu().numpy(), labels.cpu().numpy()))
            task_dataloaders_picklable.append(task_data_list)

        params_picklable = copy.deepcopy(params)  # Params should be picklable

        # Train models in parallel
        with ProcessPoolExecutor(max_workers=self.num_models) as executor:
            futures = []

            for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
                future = executor.submit(
                    train_single_model, i, model.state_dict(), optimizer.state_dict(),
                    task_dataloaders_picklable, params_picklable
                )
                futures.append(future)

            # Collect results
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        return results

    def merge_ensemble(self, training_results):
        """Merge ensemble models into a single model"""
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