# Code written by Claude 4
# Edited by Abigail Dodd

import threading
import time
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import copy


class SimpleParallelTAGem:
    """
    Simplified parallel implementation using threading instead of multiprocessing
    Better compatibility with PyTorch and complex objects
    """

    def __init__(self, model, criterion, optimizer, clustering_memory, num_threads=4):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.clustering_memory = clustering_memory
        self.num_threads = num_threads
        self.lock = threading.Lock()

    def threaded_memory_update(self, batch_data, batch_labels):
        """
        Use threading to parallelize memory updates
        """

        def update_memory_subset(start_idx, end_idx):
            """Update memory for a subset of the batch"""
            for i in range(start_idx, end_idx):
                with self.lock:  # Ensure thread-safe access to clustering_memory
                    self.clustering_memory.add_sample(batch_data[i], batch_labels[i])

        # For small batches, process sequentially
        batch_size = len(batch_data)
        if batch_size < self.num_threads * 2:
            for i in range(batch_size):
                self.clustering_memory.add_sample(batch_data[i], batch_labels[i])
            return

        # Split batch into chunks for parallel processing
        chunk_size = max(1, batch_size // self.num_threads)
        threads = []

        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            thread = threading.Thread(target=update_memory_subset, args=(i, end_idx))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

    def parallel_batch_processing(self, batch_data, batch_labels, agem_handler):
        """
        Process different parts of a batch in parallel using threading
        """
        batch_size = len(batch_data)
        if batch_size < self.num_threads:
            # Process sequentially for small batches
            return agem_handler.optimize(batch_data, batch_labels,
                                         self.clustering_memory.get_memory_samples())

        # Split batch into chunks
        chunk_size = max(1, batch_size // self.num_threads)
        chunks = []
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            chunks.append((batch_data[i:end_idx], batch_labels[i:end_idx]))

        # Process chunks in parallel
        losses = []
        memory_samples = self.clustering_memory.get_memory_samples()

        def process_chunk(chunk_data, chunk_labels):
            try:
                loss = agem_handler.calculate_loss_only(chunk_data, chunk_labels, memory_samples)
                return loss
            except Exception as e:
                print(f"Error processing chunk: {e}")
                return None

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_chunk = {
                executor.submit(process_chunk, chunk_data, chunk_labels): (chunk_data, chunk_labels)
                for chunk_data, chunk_labels in chunks
            }

            for future in as_completed(future_to_chunk):
                loss = future.result()
                if loss is not None:
                    losses.append(loss)

        # Return average loss or None if no valid losses
        return np.mean(losses) if losses else None


class SimpleParallelTrainingManager:
    """
    Simplified parallel training manager using threading
    """

    def __init__(self, params):
        self.params = params
        self.num_threads = params.get('num_threads', 4)

    def run_simple_parallel_training(self, model, optimizer, criterion, clustering_memory,
                                     task_dataloaders, agem_handler, visualizer):
        """
        Run parallel training with threading-based approach
        """
        parallel_tagem = SimpleParallelTAGem(
            model, criterion, optimizer, clustering_memory, self.num_threads
        )

        print(f"Starting simple parallel TA-A-GEM training with {self.num_threads} threads...")

        for task_id, task_dataloader in enumerate(task_dataloaders):
            print(f"\n--- Training on Task {task_id} (Simple Parallel Mode) ---")
            task_start_time = time.time()
            task_epoch_losses = []

            for epoch in range(self.params['num_epochs']):
                model.train()
                epoch_losses = []

                for batch_idx, (data, labels) in enumerate(task_dataloader):
                    # Option 1: Standard A-GEM with parallel memory updates
                    batch_loss = agem_handler.optimize(
                        data, labels, clustering_memory.get_memory_samples()
                    )

                    if batch_loss is not None:
                        epoch_losses.append(batch_loss)
                        visualizer.add_batch_loss(task_id, epoch, batch_idx, batch_loss)

                    # Parallel memory update
                    parallel_tagem.threaded_memory_update(data, labels)

                    # Progress tracking
                    if not self.params.get('quick_test_mode', False):
                        if batch_idx % 50 == 0 or batch_idx == len(task_dataloader) - 1:
                            progress = (batch_idx + 1) / len(task_dataloader)
                            bar_length = 30
                            filled_length = int(bar_length * progress)
                            bar = '█' * filled_length + '-' * (bar_length - filled_length)
                            print(
                                f'\rTask {task_id:1}, Epoch {epoch + 1:>2}/{self.params["num_epochs"]}: |{bar}| {progress:.1%} ({batch_idx + 1}/{len(task_dataloader)})',
                                end='', flush=True)

                if not self.params.get('quick_test_mode', False):
                    print()

                # Track epoch loss
                if epoch_losses:
                    avg_epoch_loss = np.mean(epoch_losses)
                    task_epoch_losses.append(avg_epoch_loss)

                    if self.params.get('quick_test_mode', False) and (
                            epoch % 5 == 0 or epoch == self.params['num_epochs'] - 1):
                        print(f'  Epoch {epoch + 1}/{self.params["num_epochs"]}: Loss = {avg_epoch_loss:.4f}')

            # Evaluation after each task
            self._evaluate_and_update_metrics(
                model, criterion, task_dataloaders, task_id, task_epoch_losses,
                clustering_memory, visualizer, time.time() - task_start_time
            )

    def _evaluate_and_update_metrics(self, model, criterion, task_dataloaders, task_id,
                                     task_losses, clustering_memory, visualizer, task_time):
        """Evaluate and update metrics after task completion"""
        # Import here to avoid circular imports
        import agem

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


# Utility functions
def run_simple_parallel_ta_agem(params, model, optimizer, criterion, clustering_memory,
                                task_dataloaders, agem_handler, visualizer):
    """
    Simple parallel training entry point
    """
    manager = SimpleParallelTrainingManager(params)
    return manager.run_simple_parallel_training(
        model, optimizer, criterion, clustering_memory,
        task_dataloaders, agem_handler, visualizer
    )
