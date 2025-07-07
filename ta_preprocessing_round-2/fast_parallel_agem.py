"""
Written by Claude 4 Sonnet
Edited by Abigail Dodd
"""

import torch
import torch.nn as nn
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import time
from queue import Queue
import multiprocessing as mp


class FastParallelTAAGEM:
    def __init__(self, model, criterion, optimizer, clustering_memory,
                 num_workers=None, use_threading=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.clustering_memory = clustering_memory
        self.num_workers = num_workers or min(8, mp.cpu_count())  # Cap at 8 for efficiency
        self.use_threading = use_threading  # Threading is often better for this use case

        # Thread-safe queues for batch processing
        self.batch_queue = Queue()
        self.result_queue = Queue()
        self.memory_update_queue = Queue()

        # Performance monitoring
        self.timing_stats = {
            'forward_pass': [],
            'backward_pass': [],
            'memory_update': [],
            'gradient_projection': []
        }

    def parallel_gradient_computation(self, batch_data, batch_labels, memory_samples):
        """
        Compute gradients in parallel for current batch and memory samples
        This is where we can actually get speedup
        """
        if len(batch_data) < 4:  # Too small for parallelization
            return self._sequential_gradient_computation(batch_data, batch_labels, memory_samples)

        # Split batch into chunks for parallel processing
        chunk_size = max(1, len(batch_data) // self.num_workers)
        chunks = []

        for i in range(0, len(batch_data), chunk_size):
            end_idx = min(i + chunk_size, len(batch_data))
            chunks.append((batch_data[i:end_idx], batch_labels[i:end_idx]))

        # Process chunks in parallel
        if self.use_threading:
            return self._threaded_gradient_computation(chunks, memory_samples)
        else:
            return self._process_gradient_computation(chunks, memory_samples)

    def _threaded_gradient_computation(self, chunks, memory_samples):
        """Use threading for gradient computation (better for PyTorch)"""
        all_gradients = []
        all_losses = []

        def compute_chunk_gradients(chunk_data, chunk_labels):
            chunk_gradients = []
            chunk_losses = []

            # Get model device
            device = next(self.model.parameters()).device

            for data, labels in zip(chunk_data, chunk_labels):
                # Ensure proper tensor shapes
                if len(data.shape) == 1:
                    data = data.unsqueeze(0)
                if len(labels.shape) == 0:
                    labels = labels.unsqueeze(0)

                # Move to correct device
                data = data.to(device)
                labels = labels.to(device)

                try:
                    # Clear gradients
                    self.optimizer.zero_grad()

                    # Forward pass - no need for data to require grad
                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)

                    # Backward pass - this computes gradients w.r.t. model parameters
                    loss.backward()

                    # Store gradients from model parameters
                    current_grads = []
                    for param in self.model.parameters():
                        if param.grad is not None:
                            current_grads.append(param.grad.clone().detach())
                        else:
                            current_grads.append(torch.zeros_like(param))

                    chunk_gradients.append(current_grads)
                    chunk_losses.append(loss.item())

                except Exception as e:
                    print(f"Warning: Error computing gradients for sample: {e}")
                    # Create zero gradients as fallback
                    zero_grads = []
                    for param in self.model.parameters():
                        zero_grads.append(torch.zeros_like(param))
                    chunk_gradients.append(zero_grads)
                    chunk_losses.append(0.0)

            return chunk_gradients, chunk_losses

        # Use ThreadPoolExecutor for better PyTorch compatibility
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for chunk_data, chunk_labels in chunks:
                future = executor.submit(compute_chunk_gradients, chunk_data, chunk_labels)
                futures.append(future)

            # Collect results
            for future in futures:
                try:
                    chunk_grads, chunk_losses = future.result(timeout=30)  # Add timeout
                    all_gradients.extend(chunk_grads)
                    all_losses.extend(chunk_losses)
                except Exception as e:
                    print(f"Warning: Thread failed: {e}")
                    # Continue with other threads

        return all_gradients, all_losses

    def _sequential_gradient_computation(self, batch_data, batch_labels, memory_samples):
        """Fallback sequential computation"""
        gradients = []
        losses = []

        for data, labels in zip(batch_data, batch_labels):
            if len(data.shape) == 1:
                data = data.unsqueeze(0)
            if len(labels.shape) == 0:
                labels = labels.unsqueeze(0)

            outputs = self.model(data)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()

            current_grads = []
            for param in self.model.parameters():
                if param.grad is not None:
                    current_grads.append(param.grad.clone())
                else:
                    current_grads.append(torch.zeros_like(param))

            gradients.append(current_grads)
            losses.append(loss.item())

        return gradients, losses

    def parallel_memory_clustering(self, batch_data, batch_labels):
        """
        Parallel clustering for memory updates
        This can be parallelized since clustering operations are independent
        """
        if len(batch_data) < self.num_workers * 2:
            # Sequential for small batches
            for data, label in zip(batch_data, batch_labels):
                self.clustering_memory.add_sample(data, label)
            return

        # Split data for parallel clustering
        chunk_size = len(batch_data) // self.num_workers
        clustering_tasks = []

        for i in range(0, len(batch_data), chunk_size):
            end_idx = min(i + chunk_size, len(batch_data))
            clustering_tasks.append((batch_data[i:end_idx], batch_labels[i:end_idx]))

        # Process clustering in parallel
        def cluster_chunk(data_chunk, label_chunk):
            local_updates = []
            for data, label in zip(data_chunk, label_chunk):
                # Pre-compute clustering info instead of directly updating
                local_updates.append((data.cpu().numpy() if torch.is_tensor(data) else data,
                                      label.cpu().numpy() if torch.is_tensor(label) else label))
            return local_updates

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(cluster_chunk, data_chunk, label_chunk)
                       for data_chunk, label_chunk in clustering_tasks]

            # Collect and apply updates sequentially (to maintain consistency)
            for future in futures:
                updates = future.result()
                for data, label in updates:
                    self.clustering_memory.add_sample(data, label)

    def optimized_training_step(self, batch_data, batch_labels):
        """
        Single optimized training step that parallelizes the right components
        """
        start_time = time.time()

        # Get current memory samples
        memory_samples = self.clustering_memory.get_memory_samples()

        # 1. Parallel gradient computation (main bottleneck)
        grad_start = time.time()
        gradients, losses = self.parallel_gradient_computation(batch_data, batch_labels, memory_samples)
        self.timing_stats['gradient_computation'] = time.time() - grad_start

        # 2. Apply A-GEM gradient projection (this needs to be sequential)
        proj_start = time.time()
        if memory_samples:
            # Compute reference gradients from memory
            ref_gradients = self._compute_reference_gradients(memory_samples)
            # Apply gradient projection
            projected_gradients = self._project_gradients(gradients, ref_gradients)
        else:
            projected_gradients = gradients

        # Apply averaged gradients
        self._apply_averaged_gradients(projected_gradients)
        self.timing_stats['gradient_projection'] = time.time() - proj_start

        # 3. Parallel memory updates
        memory_start = time.time()
        self.parallel_memory_clustering(batch_data, batch_labels)
        self.timing_stats['memory_update'] = time.time() - memory_start

        return np.mean(losses) if losses else 0.0

    def _compute_reference_gradients(self, memory_samples):
        """Compute gradients from memory samples for A-GEM projection"""
        if not memory_samples:
            return None

        # Sample a subset if memory is too large
        if len(memory_samples) > 100:  # Limit for efficiency
            indices = np.random.choice(len(memory_samples), 100, replace=False)
            memory_samples = [memory_samples[i] for i in indices]

        ref_gradients = []
        self.optimizer.zero_grad()

        total_loss = 0
        for data, label in memory_samples:
            if torch.is_tensor(data):
                data = data.unsqueeze(0) if len(data.shape) == 1 else data
            if torch.is_tensor(label):
                label = label.unsqueeze(0) if len(label.shape) == 0 else label

            outputs = self.model(data)
            loss = self.criterion(outputs, label)
            total_loss += loss

        if total_loss > 0:
            total_loss.backward()
            for param in self.model.parameters():
                if param.grad is not None:
                    ref_gradients.append(param.grad.clone())

        return ref_gradients

    def _project_gradients(self, gradients, ref_gradients):
        """Apply A-GEM gradient projection"""
        if not ref_gradients:
            return gradients

        # Average the batch gradients
        avg_gradients = []
        for param_idx in range(len(gradients[0])):
            param_grads = [grad[param_idx] for grad in gradients]
            avg_grad = torch.stack(param_grads).mean(dim=0)
            avg_gradients.append(avg_grad)

        # Project using A-GEM logic
        projected_gradients = []
        for avg_grad, ref_grad in zip(avg_gradients, ref_gradients):
            # Simple projection: if dot product is negative, project
            dot_product = torch.sum(avg_grad * ref_grad)
            if dot_product < 0:
                ref_norm_sq = torch.sum(ref_grad * ref_grad)
                if ref_norm_sq > 0:
                    projected_grad = avg_grad - (dot_product / ref_norm_sq) * ref_grad
                    projected_gradients.append(projected_grad)
                else:
                    projected_gradients.append(avg_grad)
            else:
                projected_gradients.append(avg_grad)

        return [projected_gradients]  # Return as list for consistency

    def _apply_averaged_gradients(self, gradients):
        """Apply the final gradients to the model"""
        if not gradients:
            return

        # Average all gradients
        final_gradients = []
        for param_idx in range(len(gradients[0])):
            param_grads = [grad[param_idx] for grad in gradients]
            if len(param_grads) > 1:
                final_grad = torch.stack(param_grads).mean(dim=0)
            else:
                final_grad = param_grads[0]
            final_gradients.append(final_grad)

        # Apply to model parameters
        for param, grad in zip(self.model.parameters(), final_gradients):
            param.grad = grad

        self.optimizer.step()

    def get_performance_stats(self):
        """Get performance statistics"""
        stats = {}
        for key, times in self.timing_stats.items():
            if times:
                stats[key] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'total': np.sum(times)
                }
        return stats


def run_fast_parallel_ta_agem(params, model, optimizer, criterion, clustering_memory,
                              task_dataloaders, agem_handler, visualizer):
    """
    Fast parallel training that should actually be faster than sequential
    """

    # Initialize fast parallel trainer
    fast_parallel = FastParallelTAAGEM(
        model, criterion, optimizer, clustering_memory,
        num_workers=params.get('num_processes', 4),
        use_threading=True  # Threading is usually better for PyTorch
    )

    print(f"Starting FAST parallel TA-A-GEM training with {fast_parallel.num_workers} workers...")

    overall_start_time = time.time()

    for task_id, task_dataloader in enumerate(task_dataloaders):
        print(f"\n--- Training on Task {task_id} (Fast Parallel) ---")
        task_start_time = time.time()
        task_losses = []

        for epoch in range(params['num_epochs']):
            epoch_losses = []
            epoch_start_time = time.time()

            for batch_idx, (data, labels) in enumerate(task_dataloader):
                # Use optimized parallel training step
                batch_loss = fast_parallel.optimized_training_step(data, labels)

                if batch_loss > 0:
                    epoch_losses.append(batch_loss)
                    visualizer.add_batch_loss(task_id, epoch, batch_idx, batch_loss)

                # Progress reporting
                if batch_idx % 20 == 0:
                    print(f"  Epoch {epoch + 1}/{params['num_epochs']}, "
                          f"Batch {batch_idx + 1}/{len(task_dataloader)}, "
                          f"Loss: {batch_loss:.4f}")

            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            task_losses.append(avg_epoch_loss)

            print(f"  Epoch {epoch + 1} completed in {epoch_time:.2f}s, "
                  f"Avg Loss: {avg_epoch_loss:.4f}")

        # Task evaluation (same as original)
        model.eval()
        import agem
        avg_accuracy = agem.evaluate_tasks_up_to(model, criterion, task_dataloaders, task_id)

        individual_accuracies = []
        for eval_task_id in range(task_id + 1):
            eval_dataloader = task_dataloaders[eval_task_id]
            task_acc = agem.evaluate_single_task(model, criterion, eval_dataloader)
            individual_accuracies.append(task_acc)

        task_time = time.time() - task_start_time
        memory_size = clustering_memory.get_memory_size()

        visualizer.update_metrics(
            task_id=task_id,
            overall_accuracy=avg_accuracy,
            individual_accuracies=individual_accuracies,
            epoch_losses=task_losses,
            memory_size=memory_size,
            training_time=task_time
        )

        print(f"Task {task_id} completed in {task_time:.2f}s")
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"Memory Size: {memory_size}")

        # Show performance stats
        perf_stats = fast_parallel.get_performance_stats()
        if perf_stats:
            print("Performance breakdown:")
            for operation, stats in perf_stats.items():
                print(f"  {operation}: {stats['mean']:.4f}s avg, {stats['total']:.2f}s total")

    total_time = time.time() - overall_start_time
    print(f"\nFast parallel training completed in {total_time:.2f}s")

    return fast_parallel.get_performance_stats()


# Benchmark function to compare sequential vs parallel
def benchmark_parallel_vs_sequential(params, model, optimizer, criterion, clustering_memory,
                                     task_dataloaders, agem_handler, visualizer, num_runs=1):
    """
    Benchmark parallel vs sequential performance
    """
    print("=" * 60)
    print("BENCHMARKING PARALLEL VS SEQUENTIAL")
    print("=" * 60)

    results = {'sequential': [], 'parallel': []}

    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")

        # Test sequential (simplified version)
        print("Testing sequential...")
        seq_start = time.time()

        # Simple sequential benchmark
        for task_id, task_dataloader in enumerate(task_dataloaders[:1]):  # Just first task
            for epoch in range(min(2, params['num_epochs'])):  # Just 2 epochs
                for batch_idx, (data, labels) in enumerate(task_dataloader):
                    if batch_idx >= 10:  # Just 10 batches
                        break

                    batch_loss = agem_handler.optimize(data, labels,
                                                       clustering_memory.get_memory_samples())

                    for i in range(len(data)):
                        clustering_memory.add_sample(data[i], labels[i])

        seq_time = time.time() - seq_start
        results['sequential'].append(seq_time)
        print(f"Sequential time: {seq_time:.2f}s")

        # Test parallel
        print("Testing parallel...")
        par_start = time.time()

        fast_parallel = FastParallelTAAGEM(model, criterion, optimizer, clustering_memory)

        for task_id, task_dataloader in enumerate(task_dataloaders[:1]):  # Just first task
            for epoch in range(min(2, params['num_epochs'])):  # Just 2 epochs
                for batch_idx, (data, labels) in enumerate(task_dataloader):
                    if batch_idx >= 10:  # Just 10 batches
                        break

                    batch_loss = fast_parallel.optimized_training_step(data, labels)

        par_time = time.time() - par_start
        results['parallel'].append(par_time)
        print(f"Parallel time: {par_time:.2f}s")

        speedup = seq_time / par_time if par_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x")

    # Summary
    avg_seq = np.mean(results['sequential'])
    avg_par = np.mean(results['parallel'])
    avg_speedup = avg_seq / avg_par if avg_par > 0 else 0

    print(f"\nBenchmark Results ({num_runs} runs):")
    print(f"Average Sequential: {avg_seq:.2f}s")
    print(f"Average Parallel: {avg_par:.2f}s")
    print(f"Average Speedup: {avg_speedup:.2f}x")

    if avg_speedup < 1.0:
        print("WARNING: Parallel is slower than sequential!")
        print("This might be due to small batch sizes or overhead.")

    return results
