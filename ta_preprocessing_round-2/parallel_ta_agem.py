# Code written by Claude 4
# Edited by Abigail Dodd

import multiprocessing as mp
from multiprocessing import Pool, Manager, Queue
import torch
import torch.nn as nn
import copy
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


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

        def update_memory_chunk(args):
            start_idx, end_idx, data_chunk, labels_chunk = args
            # Create a copy of clustering memory for this process
            local_updates = []

            for i in range(len(data_chunk)):
                sample_data = data_chunk[i]
                sample_label = labels_chunk[i]
                # Store update operations instead of directly modifying memory
                local_updates.append((sample_data, sample_label))

            return local_updates

        # Split batch into chunks for parallel processing
        chunk_size = max(1, len(batch_data) // self.num_processes)
        chunks = []

        for i in range(0, len(batch_data), chunk_size):
            end_idx = min(i + chunk_size, len(batch_data))
            chunks.append((i, end_idx, batch_data[i:end_idx], batch_labels[i:end_idx]))

        # Process chunks in parallel
        with Pool(processes=min(self.num_processes, len(chunks))) as pool:
            results = pool.map(update_memory_chunk, chunks)

        # Apply updates sequentially to maintain consistency
        with self.lock:
            for chunk_updates in results:
                for sample_data, sample_label in chunk_updates:
                    self.clustering_memory.add_sample(sample_data, sample_label)

    def parallel_pool_training(self, task_dataloader, task_id, num_epochs):
        """
        Parallelize training across different memory pools
        """

        def train_pool_subset(args):
            pool_id, pool_data, model_state = args

            # Create a copy of the model for this pool
            local_model = copy.deepcopy(self.model)
            local_model.load_state_dict(model_state)
            local_optimizer = torch.optim.SGD(local_model.parameters(), lr=self.optimizer.param_groups[0]['lr'])

            pool_losses = []

            # Train on this pool's data
            for data, labels in pool_data:
                local_optimizer.zero_grad()
                outputs = local_model(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                local_optimizer.step()
                pool_losses.append(loss.item())

            return pool_id, local_model.state_dict(), np.mean(pool_losses)

        # Get current model state
        current_state = self.model.state_dict()

        # Group data by pools (assuming clustering_memory has pool structure)
        pool_data = self.clustering_memory.get_pool_data()

        # Prepare arguments for parallel processing
        pool_args = [(pool_id, data, current_state) for pool_id, data in pool_data.items()]

        # Train pools in parallel
        with Pool(processes=min(self.num_processes, len(pool_args))) as pool:
            results = pool.map(train_pool_subset, pool_args)

        # Aggregate results (could use various strategies like averaging, voting, etc.)
        return self.aggregate_pool_results(results)

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


# Utility functions for integration
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
