import torch
from torch.optim.optimizer import Optimizer


class BGD(Optimizer):
    """Implements BGD optimizer for use within BGDHandler"""
    def __init__(self, params, std_init, mean_eta=1):
        super(BGD, self).__init__(params, defaults={})
        self.std_init = std_init
        self.mean_eta = mean_eta
        
        # Initialize mu (mean_param) and sigma (std_param)
        for group in self.param_groups:
            assert len(group["params"]) == 1, "BGD optimizer does not support multiple params in a group"
            assert isinstance(group["params"][0], torch.Tensor), "BGD expect param to be a tensor"
            # We use the initialization of weights to initialize the mean.
            group["mean_param"] = group["params"][0].data.clone()
            group["std_param"] = torch.zeros_like(group["params"][0].data).add_(self.std_init)
            # Initialize eps to None - will be set during randomize_weights
            group["eps"] = None

    def randomize_weights(self, force_std=-1):
        """Randomize the weights according to N(mean, std)"""
        for group in self.param_groups:
            mean = group["mean_param"]
            std = group["std_param"]
            if force_std >= 0:
                std = std.mul(0).add(force_std)
            group["eps"] = torch.normal(torch.zeros_like(mean), 1)
            # Reparameterization trick
            group["params"][0].data.copy_(mean.add(std.mul(group["eps"])))

    def get_gradients_for_update(self, batch_size):
        """Get gradients multiplied by eps for BGD update"""
        grad_sum = None
        grad_mul_eps_sum = None
        
        for group in self.param_groups:
            # Skip if no gradient or eps not initialized
            if group["params"][0].grad is None or group["eps"] is None:
                continue
            
            grad = group["params"][0].grad.data.mul(batch_size)
            if grad_sum is None:
                grad_sum = grad.clone()
                grad_mul_eps_sum = grad.mul(group["eps"]).clone()
            else:
                grad_sum.add_(grad)
                grad_mul_eps_sum.add_(grad.mul(group["eps"]))
        
        return grad_sum, grad_mul_eps_sum

    def update_parameters(self, e_grad, e_grad_eps):
        """Update mean and std parameters using aggregated gradients"""
        for group in self.param_groups:
            mean = group["mean_param"]
            std = group["std_param"]
            
            # Update mean and STD params
            mean.add_(-std.pow(2).mul(e_grad).mul(self.mean_eta))
            sqrt_term = torch.sqrt(e_grad_eps.mul(std).div(2).pow(2).add(1)).mul(std)
            std.copy_(sqrt_term.add(-e_grad_eps.mul(std.pow(2)).div(2)))


class BGDHandler:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion,
        optimizer_params: dict,
        device,
        mc_iters=10,
        lr_scheduler=None,
    ):
        """
        BGD Handler 
        
        Args:
            model: PyTorch model
            criterion: Loss function
            optimizer_params: Dict with 'std_init' and optionally 'mean_eta'
            device: Device to run on
            mc_iters: Number of Monte Carlo iterations
            lr_scheduler: Optional learning rate scheduler
        """
        self.model = model
        self.criterion = criterion
        self.device = device
        self.mc_iters = mc_iters
        self.lr_scheduler = lr_scheduler
        
        # Create BGD optimizer for each parameter group
        self.bgd_optimizers = []
        for param in model.parameters():
            if param.requires_grad:
                bgd_opt = BGD(
                    [param], 
                    std_init=optimizer_params.get('std_init', 0.1),
                    mean_eta=optimizer_params.get('mean_eta', 1.0)
                )
                self.bgd_optimizers.append(bgd_opt)

    def _get_batch_size(self, data):
        """Helper to get batch size from data"""
        if isinstance(data, torch.Tensor):
            return data.size(0)
        elif isinstance(data, (list, tuple)):
            return len(data)
        else:
            return 1

    def optimize(self, data, labels, batch_samples=None, **kwargs):
        """
        BGD optimization following the Monte Carlo procedure
        
        Args:
            data: Input data
            labels: Target labels
            batch_samples: Additional samples for training (e.g., memory replay, auxiliary data)
            **kwargs: Additional arguments (for compatibility )
        
        Returns:
            Average loss over Monte Carlo iterations
        """
        # Move data to device
        data, labels = data.to(self.device), labels.to(self.device)
        batch_size = self._get_batch_size(data)
        
        # Update learning rate if scheduler is provided
        if self.lr_scheduler is not None:
            # Compute current loss for scheduler
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(data)
                current_loss = self.criterion(outputs, labels).item()
            self.model.train()
            
            new_lr = self.lr_scheduler.step(current_loss)
            # Note: BGD doesn't use traditional learning rates, but we keep this for compatibility

        # Initialize accumulators
        total_grad_sum = {}
        total_grad_mul_eps_sum = {}
        total_loss = 0.0
        
        # Prepare all training data (main + batch samples)
        all_data_batches = [(data, labels)]
        
        # Process batch_samples if provided
        if batch_samples is not None and len(batch_samples) > 0:
            try:
                batch_data_list = []
                batch_labels_list = []
                
                for sample_item in batch_samples:
                    if isinstance(sample_item, (list, tuple)) and len(sample_item) >= 2:
                        sample_data, sample_label = sample_item[0], sample_item[1]
                    else:
                        # Assume it's just data if not tuple/list
                        sample_data, sample_label = sample_item, 0
                    
                    batch_data_list.append(sample_data)
                    batch_labels_list.append(sample_label)
                
                if batch_data_list:
                    batch_data = torch.stack(batch_data_list).to(self.device)
                    batch_labels = (
                        torch.stack(batch_labels_list).to(self.device)
                        if isinstance(batch_labels_list[0], torch.Tensor)
                        else torch.tensor(batch_labels_list).to(self.device)
                    )
                    all_data_batches.append((batch_data, batch_labels))
                    
            except Exception as e:
                print(f"Batch samples processing failed: {e}")
                # Continue with just the main data

        # Monte Carlo iterations
        for mc_iter in range(self.mc_iters):
            # Clear gradients first
            self.model.zero_grad()
            
            # Randomize weights for all BGD optimizers
            for bgd_opt in self.bgd_optimizers:
                bgd_opt.randomize_weights()
            
            # Process all data batches in this MC iteration
            mc_loss = 0.0
            for batch_data, batch_labels in all_data_batches:
                # Forward pass
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                mc_loss += loss.item()
                
                # Backward pass (accumulate gradients)
                loss.backward()
            
            total_loss += mc_loss
            
            # Aggregate gradients for each optimizer after processing all batches
            current_batch_size = sum(self._get_batch_size(bd) for bd, _ in all_data_batches)
            for i, bgd_opt in enumerate(self.bgd_optimizers):
                grad_sum, grad_mul_eps_sum = bgd_opt.get_gradients_for_update(current_batch_size)
                
                # Only accumulate if we got valid gradients
                if grad_sum is not None and grad_mul_eps_sum is not None:
                    if i not in total_grad_sum:
                        total_grad_sum[i] = grad_sum.clone()
                        total_grad_mul_eps_sum[i] = grad_mul_eps_sum.clone()
                    else:
                        total_grad_sum[i].add_(grad_sum)
                        total_grad_mul_eps_sum[i].add_(grad_mul_eps_sum)
        
        # Update parameters using aggregated gradients
        for i, bgd_opt in enumerate(self.bgd_optimizers):
            if i in total_grad_sum:
                e_grad = total_grad_sum[i].div(self.mc_iters)
                e_grad_eps = total_grad_mul_eps_sum[i].div(self.mc_iters)
                bgd_opt.update_parameters(e_grad, e_grad_eps)
        
        # Set weights to mean (force_std=0)
        for bgd_opt in self.bgd_optimizers:
            bgd_opt.randomize_weights(force_std=0)
        
        return total_loss / self.mc_iters