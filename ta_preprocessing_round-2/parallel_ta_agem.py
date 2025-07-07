# Code written by Claude 4
# Edited by Abigail Dodd

import multiprocessing as mp
from multiprocessing import Pool, Manager, Queue
import torch
import torch.nn as nn
import torch.multiprocessing as torch_mp
import copy
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import sys
from tqdm import tqdm

# Set multiprocessing start method for PyTorch compatibility
try:
    torch_mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set


# Top-level functions for multiprocessing (required for pickling)
def update_memory_chunk_worker(args):
    """Worker function for parallel memory updates - must be at module level for pickling"""
    start_idx, end_idx, data_chunk, labels_chunk = args
    local_updates = []

    # Convert tensors to CPU and detach if needed
    if torch.is_tensor(data_chunk):
        data_chunk = data_chunk.cpu().detach()
    if torch.is_tensor(labels_chunk):
        labels_chunk = labels_chunk.cpu().detach()

    for i in range(len(data_chunk)):
        sample_data = data_chunk[i]
        sample_label = labels_chunk[i]
        # Store update operations instead of directly modifying memory
        local_updates.append((sample_data, sample_label))

    return local_updates


def train_pool_subset_worker(args):
    """Worker function for parallel pool training - must be at module level for pickling"""
    pool_id, pool_data, model_state_dict, learning_rate = args

    # Reconstruct model architecture (you may need to adjust this based on your model)
    # For now, we'll assume a simple MLP structure
    local_model = create_model_from_state(model_state_dict)
    local_optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    pool_losses = []

    # Train on this pool's data
    for data, labels in pool_data:
        if torch.is_tensor(data):
            data = data.cpu()
        if torch.is_tensor(labels):
            labels = labels.cpu()

        local_optimizer.zero_grad()
        outputs = local_model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        local_optimizer.step()
        pool_losses.append(loss.item())

    return pool_id, local_model.state_dict(), np.mean(pool_losses) if pool_losses else 0.0


def create_model_from_state(state_dict):
    """Helper function to recreate model from state dict"""
    # This is a simplified version - you may need to adjust based on your model architecture
    # Extract dimensions from state dict
    input_dim = state_dict['fc1.weight'].shape[1]
    hidden_dim = state_dict['fc1.weight'].shape[0]
    num_classes = state_dict['fc3.weight'].shape[0]

    from main import SimpleMLP  # Import your model class
    model = SimpleMLP(input_dim, hidden_dim, num_classes)
    model.load_state_dict(state_dict)
    return model


class ParallelTAGem:
    """
    Parallel implementation of TA-A-GEM with multiple parallelization strategies
    """

    def __init__(self, model, criterion, optimizer, clustering_memory, num_processes=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.clustering_memory = clustering_memory
        self.num_processes = num_processes or mp.cpu_count()

        # Thread-safe components
        self.lock = threading.Lock()

    def parallel_memory_update(self, batch_data, batch_labels):
        """
        Parallelize memory updates across multiple processes
        """
        # Convert to CPU tensors for multiprocessing compatibility
        if torch.is_tensor(batch_data):
            batch_data = batch_data.cpu().detach()
        if torch.is_tensor(batch_labels):
            batch_labels = batch_labels.cpu().detach()

        # For small batches, just process sequentially to avoid overhead
        if len(batch_data) < self.num_processes * 2:
            with self.lock:
                for i in range(len(batch_data)):
                    self.clustering_memory.add_sample(batch_data[i], batch_labels[i])
            return

        # Split batch into chunks for parallel processing
        chunk_size = max(1, len(batch_data) // self.num_processes)
        chunks = []

        for i in range(0, len(batch_data), chunk_size):
            end_idx = min(i + chunk_size, len(batch_data))
            chunks.append((i, end_idx, batch_data[i:end_idx], batch_labels[i:end_idx]))

        # Process chunks in parallel
        try:
            with Pool(processes=min(self.num_processes, len(chunks))) as pool:
                results = pool.map(update_memory_chunk_worker, chunks)

            # Apply updates sequentially to maintain consistency
            with self.lock:
                for chunk_updates in results:
                    for sample_data, sample_label in chunk_updates:
                        self.clustering_memory.add_sample(sample_data, sample_label)
        except Exception as e:
            print(f"Parallel processing failed, falling back to sequential: {e}")
            # Fallback to sequential processing
            with self.lock:
                for i in range(len(batch_data)):
                    self.clustering_memory.add_sample(batch_data[i], batch_labels[i])

    def parallel_pool_training(self, task_dataloader, task_id, num_epochs):
        """
        Parallelize training across different memory pools
        """
        # Get current model state
        current_state = self.model.state_dict()
        learning_rate = self.optimizer.param_groups[0]['lr']

        # Get pool data - this assumes clustering_memory has a method to get pool data
        try:
            pool_data = self.clustering_memory.get_pool_data()
        except AttributeError:
            print("Warning: clustering_memory doesn't have get_pool_data method, using sequential training")
            return None

        if not pool_data:
            print("No pool data available for parallel training")
            return None

        # Prepare arguments for parallel processing
        pool_args = [(pool_id, data, current_state, learning_rate) for pool_id, data in pool_data.items()]

        # Train pools in parallel
        try:
            with Pool(processes=min(self.num_processes, len(pool_args))) as pool:
                results = pool.map(train_pool_subset_worker, pool_args)

            # Aggregate results
            return self.aggregate_pool_results(results)
        except Exception as e:
            print(f"Parallel pool training failed: {e}")
            return None

    def aggregate_pool_results(self, pool_results):
        """
        Aggregate results from parallel pool training
        """
        # Simple averaging strategy
        if not pool_results:
            return None

        # Average the model parameters
        avg_state_dict = {}
        total_loss = 0

        for pool_id, state_dict, loss in pool_results:
            total_loss += loss

            if not avg_state_dict:
                avg_state_dict = {k: v.clone() for k, v in state_dict.items()}
            else:
                for k, v in state_dict.items():
                    avg_state_dict[k] += v

        # Average the parameters
        num_pools = len(pool_results)
        for k in avg_state_dict:
            avg_state_dict[k] /= num_pools

        # Update main model
        self.model.load_state_dict(avg_state_dict)

        return total_loss / num_pools

    def parallel_batch_processing(self, task_dataloader, task_id, num_epochs, agem_handler):
        """
        Process batches in parallel while maintaining gradient consistency
        """

        def process_batch_chunk(args):
            batch_chunk, memory_samples = args
            local_losses = []

            for data, labels in batch_chunk:
                # Calculate gradients without updating model
                loss = agem_handler.calculate_loss(data, labels, memory_samples)
                local_losses.append(loss)

            return local_losses

        all_epoch_losses = []

        for epoch in range(num_epochs):
            epoch_losses = []

            # Convert dataloader to list for chunking
            batch_list = list(task_dataloader)
            chunk_size = max(1, len(batch_list) // self.num_processes)

            # Get current memory samples
            memory_samples = self.clustering_memory.get_memory_samples()

            # Create chunks
            chunks = []
            for i in range(0, len(batch_list), chunk_size):
                chunk = batch_list[i:i + chunk_size]
                chunks.append((chunk, memory_samples))

            # Process chunks in parallel
            with ThreadPoolExecutor(max_workers=self.num_processes) as executor:
                future_to_chunk = {executor.submit(process_batch_chunk, chunk): chunk for chunk in chunks}

                for future in as_completed(future_to_chunk):
                    chunk_losses = future.result()
                    epoch_losses.extend(chunk_losses)

            all_epoch_losses.append(np.mean(epoch_losses))

        return all_epoch_losses


class ParallelTrainingManager:
    """
    High-level manager for parallel TA-A-GEM training
    """

    def __init__(self, params):
        self.params = params
        self.parallel_strategy = params.get('parallel_strategy', 'memory_update')

    def run_parallel_training(self, model, optimizer, criterion, clustering_memory,
                              task_dataloaders, agem_handler, visualizer):
        """
        Main parallel training loop with configurable strategies
        """

        # Initialize parallel components
        parallel_tagem = ParallelTAGem(model, criterion, optimizer, clustering_memory)

        print(f"Starting parallel TA-A-GEM training with {parallel_tagem.num_processes} processes...")
        print(f"Parallel strategy: {self.parallel_strategy}")

        for task_id, task_dataloader in enumerate(task_dataloaders):
            print(f"\n--- Training on Task {task_id} (Parallel Mode) ---")
            task_start_time = time.time()

            if self.parallel_strategy == 'memory_update':
                # Parallelize memory updates
                task_losses = self._parallel_memory_strategy(
                    task_dataloader, task_id, parallel_tagem, agem_handler
                )

            elif self.parallel_strategy == 'pool_training':
                # Parallelize pool training
                task_losses = self._parallel_pool_strategy(
                    task_dataloader, task_id, parallel_tagem, agem_handler
                )

            elif self.parallel_strategy == 'batch_processing':
                # Parallelize batch processing
                task_losses = self._parallel_batch_strategy(
                    task_dataloader, task_id, parallel_tagem, agem_handler
                )

            else:
                # Hybrid approach
                task_losses = self._hybrid_parallel_strategy(
                    task_dataloader, task_id, parallel_tagem, agem_handler
                )

            # Evaluation and metrics update (same as original)
            self._update_metrics(model, criterion, task_dataloaders, task_id,
                                 task_losses, clustering_memory, visualizer,
                                 time.time() - task_start_time)

    def _parallel_memory_strategy(self, task_dataloader, task_id, parallel_tagem, agem_handler):
        """Strategy focusing on parallelizing memory operations"""
        task_losses = []

        for epoch in range(self.params['num_epochs']):
            epoch_losses = []

            for batch_idx, (data, labels) in enumerate(task_dataloader):
                # Standard A-GEM optimization
                batch_loss = agem_handler.optimize(data, labels,
                                                   parallel_tagem.clustering_memory.get_memory_samples())

                if batch_loss is not None:
                    epoch_losses.append(batch_loss)

                # Parallel memory update
                parallel_tagem.parallel_memory_update(data, labels)

            if epoch_losses:
                task_losses.append(np.mean(epoch_losses))

        return task_losses

    def _parallel_pool_strategy(self, task_dataloader, task_id, parallel_tagem, agem_handler):
        """Strategy focusing on parallelizing pool training"""
        return parallel_tagem.parallel_pool_training(task_dataloader, task_id, self.params['num_epochs'])

    def _parallel_batch_strategy(self, task_dataloader, task_id, parallel_tagem, agem_handler):
        """Strategy focusing on parallelizing batch processing"""
        return parallel_tagem.parallel_batch_processing(task_dataloader, task_id,
                                                        self.params['num_epochs'], agem_handler)

    def _hybrid_parallel_strategy(self, task_dataloader, task_id, parallel_tagem, agem_handler):
        """Hybrid strategy combining multiple parallelization approaches"""
        # Combine memory updates and batch processing
        memory_losses = self._parallel_memory_strategy(task_dataloader, task_id, parallel_tagem, agem_handler)
        batch_losses = self._parallel_batch_strategy(task_dataloader, task_id, parallel_tagem, agem_handler)

        # Combine results (weighted average or other strategy)
        if len(memory_losses) == len(batch_losses):
            return [(m + b) / 2 for m, b in zip(memory_losses, batch_losses)]
        else:
            return memory_losses or batch_losses

    def _update_metrics(self, model, criterion, task_dataloaders, task_id, task_losses,
                        clustering_memory, visualizer, task_time):
        """Update metrics after parallel training (same as original)"""
        import agem  # Assuming this is imported in your main file

        model.eval()
        avg_accuracy = agem.evaluate_tasks_up_to(model, criterion, task_dataloaders, task_id)

        individual_accuracies = []
        for eval_task_id in range(task_id + 1):
            eval_dataloader = task_dataloaders[eval_task_id]
            task_acc = agem.evaluate_single_task(model, criterion, eval_dataloader)
            individual_accuracies.append(task_acc)

        memory_size = clustering_memory.get_memory_size()
        pool_sizes = clustering_memory.get_pool_sizes()
        num_active_pools = clustering_memory.get_num_active_pools()

        visualizer.update_metrics(
            task_id=task_id,
            overall_accuracy=avg_accuracy,
            individual_accuracies=individual_accuracies,
            epoch_losses=task_losses,
            memory_size=memory_size,
            training_time=task_time
        )

        print(f"After Task {task_id}, Average Accuracy: {avg_accuracy:.4f}")
        print(f"Memory Size: {memory_size} samples across {num_active_pools} active pools")
        print(f"Pool sizes: {pool_sizes}")
        print(f"Task Training Time: {task_time:.2f}s")


class ProgressTracker:
    """Thread-safe progress tracker for parallel training"""

    def __init__(self):
        self.lock = threading.Lock()
        self.current_task = 0
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        self.start_time = time.time()
        self.task_start_time = time.time()
        self.epoch_losses = []

    def start_task(self, task_id, total_batches):
        with self.lock:
            self.current_task = task_id
            self.current_epoch = 0
            self.current_batch = 0
            self.total_batches = total_batches
            self.task_start_time = time.time()
            self.epoch_losses = []
            print(f"\n--- Starting Task {task_id} ---")

    def start_epoch(self, epoch):
        with self.lock:
            self.current_epoch = epoch
            self.current_batch = 0
            print(f"Epoch {epoch + 1}")

    def update_batch(self, batch_idx, loss=None):
        with self.lock:
            self.current_batch = batch_idx
            if loss is not None:
                self.epoch_losses.append(loss)

            # Show progress every 1000 batches or at the end
            if batch_idx % 1000 == 0 or batch_idx == self.total_batches - 1:
                progress = (batch_idx + 1) / self.total_batches
                elapsed = time.time() - self.task_start_time
                eta = (elapsed / progress) - elapsed if progress > 0 else 0

                loss_str = f", Loss: {loss:.4f}" if loss is not None else ""
                print(f"  Batch {batch_idx + 1}/{self.total_batches} "
                      f"({progress:.1%}) - "
                      f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s{loss_str}")

    def finish_epoch(self):
        with self.lock:
            if self.epoch_losses:
                avg_loss = sum(self.epoch_losses) / len(self.epoch_losses)
                print(f"  Epoch {self.current_epoch + 1} completed - Avg Loss: {avg_loss:.4f}")
            self.epoch_losses = []

    def finish_task(self):
        with self.lock:
            task_time = time.time() - self.task_start_time
            print(f"Task {self.current_task} completed in {task_time:.2f}s")
            return task_time


class EnhancedParallelTrainingManager(ParallelTrainingManager):
    """Enhanced version with progress tracking and timeout detection"""

    def __init__(self, params):
        super().__init__(params)
        self.progress_tracker = ProgressTracker()
        self.timeout_seconds = params.get('timeout_seconds', 300)  # 5 minutes default

    def run_parallel_training(self, model, optimizer, criterion, clustering_memory,
                              task_dataloaders, agem_handler, visualizer):
        """Enhanced parallel training with progress tracking"""

        # Initialize parallel components
        parallel_tagem = ParallelTAGem(model, criterion, optimizer, clustering_memory)

        print(f"Starting parallel TA-A-GEM training with {parallel_tagem.num_processes} processes...")
        print(f"Parallel strategy: {self.parallel_strategy}")
        print(f"Timeout: {self.timeout_seconds}s per task")

        for task_id, task_dataloader in enumerate(task_dataloaders):
            total_batches = len(task_dataloader)
            self.progress_tracker.start_task(task_id, total_batches)

            try:
                # Set up timeout detection
                task_start_time = time.time()

                if self.parallel_strategy == 'memory_update':
                    task_losses = self._enhanced_parallel_memory_strategy(
                        task_dataloader, task_id, parallel_tagem, agem_handler
                    )
                elif self.parallel_strategy == 'pool_training':
                    task_losses = self._enhanced_parallel_pool_strategy(
                        task_dataloader, task_id, parallel_tagem, agem_handler
                    )
                else:
                    # Fallback to memory strategy with progress tracking
                    task_losses = self._enhanced_parallel_memory_strategy(
                        task_dataloader, task_id, parallel_tagem, agem_handler
                    )

                task_time = self.progress_tracker.finish_task()

                # Check if we hit timeout
                if time.time() - task_start_time > self.timeout_seconds:
                    print(f"WARNING: Task {task_id} may have timed out!")

            except KeyboardInterrupt:
                print(f"\nTraining interrupted by user during Task {task_id}")
                break
            except Exception as e:
                print(f"ERROR during Task {task_id}: {e}")
                print("Falling back to sequential training for this task...")
                task_losses = self._fallback_sequential_training(
                    task_dataloader, task_id, model, optimizer, criterion,
                    clustering_memory, agem_handler
                )
                task_time = self.progress_tracker.finish_task()

            # Update metrics
            self._update_metrics(model, criterion, task_dataloaders, task_id,
                                 task_losses, clustering_memory, visualizer, task_time)

    def _enhanced_parallel_memory_strategy(self, task_dataloader, task_id, parallel_tagem, agem_handler):
        """Memory strategy with progress tracking"""
        task_losses = []
        total_batches = len(task_dataloader)

        for epoch in range(self.params['num_epochs']):
            self.progress_tracker.start_epoch(epoch)
            epoch_losses = []

            for batch_idx, (data, labels) in enumerate(task_dataloader):
                # Check for timeout periodically
                if batch_idx % 50 == 0:  # Check every 50 batches
                    elapsed = time.time() - self.progress_tracker.task_start_time
                    if elapsed > self.timeout_seconds:
                        print(f"\nTimeout detected after {elapsed:.1f}s!")
                        raise TimeoutError(f"Task {task_id} exceeded timeout of {self.timeout_seconds}s")

                # Standard A-GEM optimization
                batch_loss = agem_handler.optimize(
                    data, labels, parallel_tagem.clustering_memory.get_memory_samples()
                )

                if batch_loss is not None:
                    epoch_losses.append(batch_loss)

                # Update progress
                self.progress_tracker.update_batch(batch_idx, batch_loss)

                # Parallel memory update
                try:
                    parallel_tagem.parallel_memory_update(data, labels)
                except Exception as e:
                    print(f"Parallel memory update failed, using sequential: {e}")
                    # Fallback to sequential memory update
                    for i in range(len(data)):
                        parallel_tagem.clustering_memory.add_sample(data[i], labels[i])

            self.progress_tracker.finish_epoch()

            if epoch_losses:
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                task_losses.append(avg_epoch_loss)

        return task_losses

    def _enhanced_parallel_pool_strategy(self, task_dataloader, task_id, parallel_tagem, agem_handler):
        """Pool strategy with progress tracking"""
        print(f"Training {parallel_tagem.clustering_memory.get_num_active_pools()} pools in parallel...")

        start_time = time.time()
        result = parallel_tagem.parallel_pool_training(
            task_dataloader, task_id, self.params['num_epochs']
        )

        elapsed = time.time() - start_time
        print(f"Pool training completed in {elapsed:.2f}s")

        return result if result is not None else []

    def _fallback_sequential_training(self, task_dataloader, task_id, model, optimizer,
                                      criterion, clustering_memory, agem_handler):
        """Fallback to sequential training if parallel fails"""
        print("Using sequential training as fallback...")
        task_losses = []

        for epoch in range(self.params['num_epochs']):
            epoch_losses = []

            for batch_idx, (data, labels) in enumerate(task_dataloader):
                batch_loss = agem_handler.optimize(data, labels, clustering_memory.get_memory_samples())

                if batch_loss is not None:
                    epoch_losses.append(batch_loss)

                # Sequential memory update
                for i in range(len(data)):
                    clustering_memory.add_sample(data[i], labels[i])

                # Simple progress indication
                if batch_idx % 50 == 0:
                    print(f"  Fallback: Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(task_dataloader)}")

            if epoch_losses:
                task_losses.append(sum(epoch_losses) / len(epoch_losses))

        return task_losses


# Utility functions
def run_enhanced_parallel_ta_agem(params, model, optimizer, criterion, clustering_memory,
                                  task_dataloaders, agem_handler, visualizer):
    """
    Enhanced parallel training with progress tracking and error handling
    """
    # Add timeout parameter if not present
    if 'timeout_seconds' not in params:
        params['timeout_seconds'] = 300  # 5 minutes default

    manager = EnhancedParallelTrainingManager(params)
    return manager.run_parallel_training(
        model, optimizer, criterion, clustering_memory,
        task_dataloaders, agem_handler, visualizer
    )


def create_parallel_params(original_params, parallel_strategy='memory_update', num_processes=None):
    """Create parameters for parallel training"""
    parallel_params = original_params.copy()
    parallel_params['parallel_strategy'] = parallel_strategy
    parallel_params['num_processes'] = num_processes or mp.cpu_count()
    return parallel_params


def run_parallel_ta_agem(params, model, optimizer, criterion, clustering_memory,
                         task_dataloaders, agem_handler, visualizer):
    """
    Drop-in replacement for the main training loop with parallel processing
    """
    manager = ParallelTrainingManager(params)
    return manager.run_parallel_training(
        model, optimizer, criterion, clustering_memory,
        task_dataloaders, agem_handler, visualizer
    )

