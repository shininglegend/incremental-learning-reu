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

    def compute_gradient(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute and return the flattened gradient vector for a single example (x, y).
        g = ∇_θ ℓ(f_θ(x, t), y)
        """
        # Move to correct device
        x, y = x.to(self.device), y.to(self.device)

        # Zero existing gradients
        self.model.zero_grad(set_to_none=True)

        # Forward + loss
        output = self.model(x.unsqueeze(0))  # add batch dim
        loss = self.criterion(output, y.view(1))

        # Collect gradients
        params = [p for p in self.model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True
        )

        # Flatten, padding zeros for unused params
        flat_grad = []
        for g, p in zip(grads, params):
            numel = p.numel()
            if g is None:
                flat_grad.append(torch.zeros(numel, device=self.device))
            else:
                flat_grad.append(g.contiguous().view(-1))
        return torch.cat(flat_grad)

    def project_gradient(self, g: torch.Tensor, g_ref: torch.Tensor) -> torch.Tensor:
        """
        A-GEM projection: g_tilde = g if no interference, else project onto non-conflicting direction.
        """
        # Empty gradients or no reference
        if g.numel() == 0 or g_ref.numel() == 0:
            return g

        # Ensure same device and shape
        assert g.shape == g_ref.shape, "Gradient shape mismatch"

        # Dot products
        dot = torch.dot(g, g_ref)
        if dot >= 0:
            return g
        norm_sq = torch.dot(g_ref, g_ref)
        if norm_sq == 0:
            return g

        # Projection
        scale = dot / norm_sq
        return g - scale * g_ref

    def set_gradient(self, flat_grad: torch.Tensor):
        """
        Unpack and assign flattened gradient vector back to model parameters.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        idx = 0
        for p in params:
            numel = p.numel()
            p.grad = flat_grad[idx: idx + numel].view_as(p).clone()
            idx += numel

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
        """Sample using global index"""

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
        print("ya done messed something up. agem.py/sample_from_memory_global_index.")
        print(f"mem size: {mem_size}. global_idx: {global_idx}")
        return None, None

    def sample_from_memory_pool(self, label, clustering_memory):

        # get the pool that corresponds to the new sample
        pool = clustering_memory.pools[label]
        if pool is None:
            print("Sampling from a pool that doesn't exist.")

        pool_size = pool.num_samples
        if pool_size == 0:
            return None, None

        # Retrieve random sample from this pool
        idx = torch.randint(0, pool_size, (1,)).item()
        current_idx = 0

        # Return the random sample
        for i, cluster in enumerate(pool.clusters):
            cluster_size = len(cluster.samples)

            # Skip empty clusters entirely
            if cluster_size == 0:
                continue

            # Check if our target index is in this cluster
            if idx < current_idx + cluster_size:
                # Found it! Get the local index within this cluster
                local_idx = idx - current_idx

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

        print("oops no sample!")
        return None, None

    def optimize_single_example(self, x, y, clustering_memory):
        """
        A-GEM optimization for a single example (x, y) following Algorithm 2, lines 6-14 in A-GEM paper
        """

        self.t.start("sample from memory")
        # Sample (x_ref, y_ref) from memory
        x_ref, y_ref = self.sample_from_memory_global_index(clustering_memory)
        # x_ref, y_ref = self.sample_from_memory_pool(clustering_memory, y)
        self.t.end("sample from memory")

        self.t.start("compute current gradient")
        # Step 8: Compute gradient
        g = self.compute_gradient(x, y)
        self.t.end("compute current gradient")

        # If we have memory samples, compute reference gradient and project
        self.t.start("compute and project gradient")
        if x_ref is not None and y_ref is not None:
            # Step 7: Compute g_ref = ∇_θ ℓ(f_θ(x_ref, t), y_ref)
            g_ref = self.compute_gradient(x_ref, y_ref)

            # Steps 9-13: Project gradient if needed
            g_tilde = self.project_gradient(g, g_ref)
        else:
            g_tilde = g
        
        # Set the projected gradient
        self.set_gradient(g_tilde)
        self.t.end("compute and project gradient")

        # Step 14: Update parameters θ ← θ - α * g_tilde
        self.t.start("optimizer.step")
        self.optimizer.step()
        self.t.end("optimizer.step")

        self.t.start("criterion")
        with torch.no_grad():
            out = self.model(x.unsqueeze(0))
            criterion = self.criterion(out, y.view(1)).item()
            self.t.end("criterion")
            return criterion

    def optimize(self, data, labels, clustering_memory):
        """
        A-GEM optimization interface that requires clustering_memory

        Args:
            data: Batch of training data
            labels: Batch of training labels
            clustering_memory: ClusteringMemory instance for sampling reference examples

        Returns:
            float: Average loss for the batch
        """
        """
                Optimize a batch by applying A-GEM to each example individually
                following the algorithm structure
                """
        batch_losses = []

        # Apply A-GEM to each example in the batch
        for i in range(data.size(0)):
            x_i = data[i]
            y_i = labels[i]
            loss = self.optimize_single_example(x_i, y_i, clustering_memory)
            batch_losses.append(loss)

        return sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
