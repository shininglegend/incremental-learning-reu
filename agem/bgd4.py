import torch
from torch.optim.optimizer import Optimizer


class BGD(Optimizer):
    """Implements BGD optimizer for use within BGDHandler"""
    def __init__(self, params, defaults=None):
        if defaults is None:
            defaults = {'std_init': 0.1, 'mean_eta': 1.0}
        super(BGD, self).__init__(params, defaults)
        
        # Initialize mu (mean_param) and sigma (std_param)
        for group in self.param_groups:
            std_init = group.get('std_init', self.defaults['std_init'])
            for param in group['params']:
                if not isinstance(param, torch.Tensor):
                    raise TypeError("BGD expects parameters to be tensors")
            
            # Store mean and std for each parameter in the group
            group['mean_params'] = []
            group['std_params'] = []
            group['eps_params'] = []
            
            for param in group['params']:
                # We use the initialization of weights to initialize the mean
                group['mean_params'].append(param.data.clone())
                group['std_params'].append(torch.zeros_like(param.data).add_(std_init))
                group['eps_params'].append(None)

    def randomize_weights(self, force_std=-1):
        """Randomize the weights according to N(mean, std)"""
        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                mean = group['mean_params'][i]
                std = group['std_params'][i]
                if force_std >= 0:
                    std = std.mul(0).add(force_std)
                group['eps_params'][i] = torch.normal(torch.zeros_like(mean), 1)
                # Reparameterization trick
                param.data.copy_(mean.add(std.mul(group['eps_params'][i])))

    def get_gradients_for_update(self, batch_size):
        """Get gradients multiplied by eps for BGD update"""
        all_grad_sums = []
        all_grad_mul_eps_sums = []
        
        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                if param.grad is None:
                    continue
                eps = group['eps_params'][i]
                if eps is None:
                    raise RuntimeError("Must randomize weights before getting gradients")
                
                grad = param.grad.data.mul(batch_size)
                all_grad_sums.append(grad)
                all_grad_mul_eps_sums.append(grad.mul(eps))
        
        return all_grad_sums, all_grad_mul_eps_sums

    def update_parameters(self, all_e_grads, all_e_grad_eps):
        """Update mean and std parameters using aggregated gradients"""
        grad_idx = 0
        for group in self.param_groups:
            mean_eta = group.get('mean_eta', self.defaults['mean_eta'])
            for i, param in enumerate(group['params']):
                if grad_idx >= len(all_e_grads):
                    break
                    
                mean = group['mean_params'][i]
                std = group['std_params'][i]
                e_grad = all_e_grads[grad_idx]
                e_grad_eps = all_e_grad_eps[grad_idx]
                
                # Update mean and STD params
                mean.add_(-std.pow(2).mul(e_grad).mul(mean_eta))
                sqrt_term = torch.sqrt(e_grad_eps.mul(std).div(2).pow(2).add(1)).mul(std)
                std.copy_(sqrt_term.add(-e_grad_eps.mul(std.pow(2)).div(2)))
                
                grad_idx += 1


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
        BGD Handler that mimics AGEMHandler structure
        
        Args:
            model: PyTorch model
            criterion: Loss function
            optimizer_params: Dict or list of dicts with BGD parameters:
                - If dict: Applied to all parameters {'std_init': 0.1, 'mean_eta': 1.0}
                - If list: Each dict applied to corresponding parameter groups
                    [{'params': model.conv_layers.parameters(), 'std_init': 0.1, 'mean_eta': 1.0},
                     {'params': model.fc_layers.parameters(), 'std_init': 0.05, 'mean_eta': 0.5}]
            device: Device to run on
            mc_iters: Number of Monte Carlo iterations
            lr_scheduler: Optional learning rate scheduler
        """
        self.model = model
        self.criterion = criterion
        self.device = device
        self.mc_iters = mc_iters
        self.lr_scheduler = lr_scheduler
        
        # Create BGD optimizer with parameter groups
        if isinstance(optimizer_params, dict):
            # Single set of params for all model parameters
            param_groups = [{'params': model.parameters(), **optimizer_params}]
        elif isinstance(optimizer_params, list):
            # Multiple parameter groups with different settings
            param_groups = optimizer_params
        else:
            raise ValueError("optimizer_params must be dict or list of dicts")
        
        self.bgd_optimizer = BGD(param_groups)

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
            **kwargs: Additional arguments (for compatibility with AGEMHandler)
        
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
        total_grad_sums = []
        total_grad_mul_eps_sums = []
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
            # Randomize weights
            self.bgd_optimizer.randomize_weights()
            
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
            
            # Aggregate gradients after processing all batches
            current_batch_size = sum(self._get_batch_size(bd) for bd, _ in all_data_batches)
            grad_sums, grad_mul_eps_sums = self.bgd_optimizer.get_gradients_for_update(current_batch_size)
            
            if len(grad_sums) > 0:
                if len(total_grad_sums) == 0:
                    total_grad_sums = [g.clone() for g in grad_sums]
                    total_grad_mul_eps_sums = [g.clone() for g in grad_mul_eps_sums]
                else:
                    for i, (g1, g2) in enumerate(zip(grad_sums, grad_mul_eps_sums)):
                        total_grad_sums[i].add_(g1)
                        total_grad_mul_eps_sums[i].add_(g2)
            
            # Clear gradients for next MC iteration
            self.model.zero_grad()
        
        # Update parameters using aggregated gradients
        if len(total_grad_sums) > 0:
            e_grads = [g.div(self.mc_iters) for g in total_grad_sums]
            e_grad_eps = [g.div(self.mc_iters) for g in total_grad_mul_eps_sums]
            self.bgd_optimizer.update_parameters(e_grads, e_grad_eps)
        
        # Set weights to mean (force_std=0)
        self.bgd_optimizer.randomize_weights(force_std=0)
        
        return total_loss / self.mc_iters