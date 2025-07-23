import torch
from visualization_analysis.visualization_analysis import Timer

class AGEMHandler:
    def __init__(
            self,
            model: torch.nn.Module,
            criterion,
            optimizer,
            device,
            batch_size=256,  # Memory batch size for gradient computation
            lr_scheduler=None,
            t=None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.eps_mem_batch = batch_size  # Memory batch size for gradient computation
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.t = t

    def compute_gradient(self, data, labels):
        """Compute gradients for given data and labels without corrupting model state"""
        # Move data to device
        data, labels = data.to(self.device), labels.to(self.device)

        # Save current gradients
        current_grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                current_grads.append(param.grad.clone())
            else:
                current_grads.append(None)

        # Compute new gradients
        self.model.zero_grad()
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        loss.backward()

        # Extract gradients
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))

        # Restore original gradients
        for param, original_grad in zip(self.model.parameters(), current_grads):
            if original_grad is not None:
                param.grad = original_grad
            else:
                param.grad = None

        return torch.cat(grads) if grads else torch.tensor([]).to(self.device)

    def project_gradient_agem(self, g, g_ref):
        """
        Project gradient g using A-GEM formula from Algorithm 2, line 12:
        g_tilde = g - (g^T * g_ref) / (g_ref^T * g_ref) * g_ref
        """
        if g_ref.numel() == 0 or g.numel() == 0:
            return g

        # Compute dot products
        g_dot_g_ref = torch.dot(g, g_ref)  # g^T * g_ref

        # If g^T * g_ref >= 0, no projection needed (line 9-10)
        if g_dot_g_ref >= 0:
            return g

        # Otherwise, apply projection (line 12)
        g_ref_dot_g_ref = torch.dot(g_ref, g_ref)  # g_ref^T * g_ref

        if g_ref_dot_g_ref > 0:
            # g_tilde = g - (g^T * g_ref) / (g_ref^T * g_ref) * g_ref
            projected_grad = g - (g_dot_g_ref / g_ref_dot_g_ref) * g_ref
            return projected_grad

        return g

    def set_gradient(self, grad_vector):
        """Set model gradients from flattened gradient vector"""
        if grad_vector.numel() == 0:
            return

        pointer = 0
        for param in self.model.parameters():
            if param.grad is not None:
                num_param = param.numel()
                param.grad.data = grad_vector[pointer: pointer + num_param].view(
                    param.shape
                )
                pointer += num_param

    def sample_from_memory(self, clustering_memory):
        """Sample (x_ref, y_ref) from ClusteringMemory as in line 6 of Algorithm 2"""
        memory_samples = clustering_memory.get_memory_samples()

        if memory_samples is None:
            return None, None

        mem_data, mem_labels = memory_samples

        if len(mem_data) == 0:
            return None, None

        # Randomly sample one example from memory
        idx = torch.randint(0, len(mem_data), (1,)).item()
        x_ref = mem_data[idx].unsqueeze(0)  # Add batch dimension
        y_ref = mem_labels[idx].unsqueeze(0)  # Add batch dimension

        return x_ref, y_ref

    def sample_from_memory_simple(self, clustering_memory):
        """Sample from a random pool, then random sample within that pool"""

        # Get non-empty pools
        non_empty_pools = []
        for label, pool in clustering_memory.pools.items():
            samples, labels = pool.get_clusters_with_labels()
            if len(samples) > 0:
                non_empty_pools.append((samples, labels if labels else [label] * len(samples)))

        if not non_empty_pools:
            return None, None

        # Pick a random pool
        pool_idx = torch.randint(0, len(non_empty_pools), (1,)).item()
        samples, labels = non_empty_pools[pool_idx]

        # Pick random sample from that pool
        sample_idx = torch.randint(0, len(samples), (1,)).item()

        x_ref = samples[sample_idx].unsqueeze(0).to(clustering_memory.device)
        y_ref = torch.tensor([labels[sample_idx]], dtype=torch.long).to(clustering_memory.device)

        return x_ref, y_ref

    def sample_from_memory_global_index(self, clustering_memory):
        """Sample using global index - O(1) complexity, no tensor operations"""

        mem_size = clustering_memory.get_memory_size()

        if mem_size == 0:
            return None, None

        # Generate random global index
        global_idx = torch.randint(0, mem_size, (1,)).item()

        # Find the sample at this global index
        current_idx = 0

        for label, pool in clustering_memory.pools.items():
            for i, cluster in enumerate(pool.clusters):
                cluster_size = len(cluster.samples)

                # Skip empty clusters entirely
                if cluster_size == 0:
                    continue

                # Check if our target index is in this cluster
                if global_idx < current_idx + cluster_size:
                    # Found it! Get the local index within this cluster
                    local_idx = global_idx - current_idx

                    # Get the sample directly from the cluster
                    try:
                        sample = list(cluster.samples)[local_idx]
                        cluster_label = cluster.labels[local_idx]

                        # Move to device and add batch dimension
                        x_ref = sample.unsqueeze(0).to(clustering_memory.device)
                        y_ref = torch.tensor([cluster_label], dtype=torch.long).to(clustering_memory.device)

                        return x_ref, y_ref
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        return None, None

                current_idx += cluster_size

        # Should never reach here if total_samples is correct
        print("ya done fucked something up. agem.py/sample_from_memory_global_index.")
        print(f"mem size: {mem_size}. global_idx: {global_idx}")
        return None, None

    def optimize_single_example(self, x, y, clustering_memory):
        """
        A-GEM optimization for a single example (x, y) following Algorithm 2, lines 6-14
        """
        # Move data to device
        x, y = x.to(self.device), y.to(self.device)

        # Step 6: Sample (x_ref, y_ref) from memory M
        self.t.start("sample from memory")
        x_ref, y_ref = self.sample_from_memory_global_index(clustering_memory)
        self.t.end("sample from memory")


        self.t.start("compute current gradient")

        # Step 8: Compute gradient g = ∇_θ ℓ(f_θ(x, t), y)
        self.model.zero_grad()
        output = self.model(x.unsqueeze(0))  # Add batch dimension
        loss = self.criterion(output, y.unsqueeze(0))
        loss.backward()

        # Extract current gradient g
        g = []
        for param in self.model.parameters():
            if param.grad is not None:
                g.append(param.grad.view(-1))
        g = torch.cat(g) if g else torch.tensor([]).to(self.device)

        self.t.end("compute current gradient")

        # If we have memory samples, compute reference gradient and project
        self.t.start("compute and project gradient")
        if x_ref is not None and y_ref is not None:
            # Step 7: Compute g_ref = ∇_θ ℓ(f_θ(x_ref, t), y_ref)
            g_ref = self.compute_gradient(x_ref, y_ref)

            # Steps 9-13: Project gradient if needed
            g_tilde = self.project_gradient_agem(g, g_ref)

            # Set the projected gradient
            self.set_gradient(g_tilde)
        self.t.end("compute and project gradient")

        # Step 14: Update parameters θ ← θ - α * g_tilde
        self.t.start("optimizer.step")
        self.optimizer.step()
        self.t.end("optimizer.step")

        return loss.item()

    def optimize_batch(self, data, labels, clustering_memory):
        """
        Optimize a batch by applying A-GEM to each example individually
        following the algorithm structure
        """
        batch_losses = []

        self.t.start("optimize batch")
        # Apply A-GEM to each example in the batch
        for i in range(data.size(0)):
            x_i = data[i]
            y_i = labels[i]
            loss = self.optimize_single_example(x_i, y_i, clustering_memory)
            batch_losses.append(loss)
        self.t.end("optimize batch")

        return sum(batch_losses) / len(batch_losses) if batch_losses else 0.0

    # Clean interface for true A-GEM
    def optimize(self, data, labels, clustering_memory):
        """
        Clean A-GEM optimization interface that requires clustering_memory

        Args:
            data: Batch of training data
            labels: Batch of training labels
            clustering_memory: ClusteringMemory instance for sampling reference examples

        Returns:
            float: Average loss for the batch
        """
        return self.optimize_batch(data, labels, clustering_memory)
