import torch
#import torch.nn.functional as F


class AGEMHandler:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion,
        optimizer,
        device,
        batch_size=256,
        lr_scheduler=None,
        epsilon=1e-3,  # Sensitivity parameter e for epsilon
        si_lambda=0.1,  # Synaptic Intelligence regularization strength
        si_xi=1.0,  # SI damping parameter
        si_update_freq=100,  # How often to consolidate importance (steps), 0 = only manual
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.eps_mem_batch = batch_size  # Memory batch size for gradient computation
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.epsilon = epsilon  # Sensitivity parameter for MEGA-I
        
        # Synaptic Intelligence parameters
        self.si_lambda = si_lambda
        self.si_xi = si_xi
        self.si_update_freq = si_update_freq
        self.si_step_count = 0  # Track steps for automatic consolidation
        
        # Initialize SI tracking variables
        self.init_synaptic_intelligence()

    def init_synaptic_intelligence(self):
        """Initialize Synaptic Intelligence tracking variables"""
        self.si_omega = {}  # Parameter importance weights
        self.si_W = {}      # Accumulated parameter changes
        self.si_prev_params = {}  # Previous parameter values
        self.prev_loss = None  # Previous loss value
        
        # Initialize tracking for all model parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.si_omega[name] = torch.zeros_like(param.data).to(self.device)
                self.si_W[name] = torch.zeros_like(param.data).to(self.device)
                self.si_prev_params[name] = param.data.clone().to(self.device)

    def update_si_tracking(self, current_loss):
        """Update Synaptic Intelligence tracking variables"""
        if self.prev_loss is not None:
            loss_diff = self.prev_loss - current_loss
            
            # Only update if loss actually decreased
            if loss_diff > 0:
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in self.si_W:
                        # Calculate parameter change
                        param_change = param.data - self.si_prev_params[name]
                        
                        # Update accumulated parameter changes weighted by loss reduction
                        if param.grad is not None:
                            # W += -grad * param_change * loss_diff
                            self.si_W[name] += -param.grad.data * param_change * loss_diff
        
        # Update previous parameters and loss
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.si_prev_params:
                self.si_prev_params[name] = param.data.clone()
        
        self.prev_loss = current_loss

    def consolidate_si_importance(self, reset_accumulation=True):
        """Consolidate parameter importance - can be called continuously or at task boundaries"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.si_omega:
                # Calculate parameter importance: omega = W / ((param_change)^2 + xi)
                param_change = param.data - self.si_prev_params[name]
                param_change_sq = param_change ** 2
                
                # Only update if there's meaningful parameter change
                if param_change_sq.sum() > 1e-8:
                    importance_update = self.si_W[name] / (param_change_sq + self.si_xi)
                    self.si_omega[name] += importance_update.abs()  # Use absolute value for importance
                
                # Optionally reset accumulated changes (for task boundaries)
                if reset_accumulation:
                    self.si_W[name] = torch.zeros_like(param.data).to(self.device)

    def compute_si_loss(self):
        """Compute Synaptic Intelligence regularization loss"""
        si_loss = 0.0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.si_omega:
                # SI loss: lambda * sum(omega * (theta - theta_prev)^2)
                param_diff = param - self.si_prev_params[name]
                si_loss += (self.si_omega[name] * param_diff ** 2).sum()
        
        return self.si_lambda * si_loss

    def compute_gradient(self, data, labels, include_si_reg=True):
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
        base_loss = self.criterion(outputs, labels)
        
        if include_si_reg:
            # Add Synaptic Intelligence regularization
            si_loss = self.compute_si_loss()
            total_loss = base_loss + si_loss
        else:
            total_loss = base_loss
        
        #base_loss.backward() # added, gives major error
        total_loss.backward() # computes all required gradients

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

        return torch.cat(grads) if grads else torch.tensor([]).to(self.device), base_loss.item()

    def compute_loss(self, data, labels):
        """Compute loss for given data and labels without affecting gradients"""
        data, labels = data.to(self.device), labels.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(data)
            base_loss = self.criterion(outputs, labels)
            si_loss = self.compute_si_loss()
            total_loss = base_loss + si_loss
        
        return base_loss.item()  # Return base loss for MEGA-I balancing

    def mega_i_gradient_balance(self, current_grad, ref_grad, current_loss, ref_loss):
        """
        Balance current and reference gradients using MEGA-I approach based on loss information.
        
        According to equation:
        - If l_t(w; X) > e, then a1(w) = 1, a2(w) = l_ref(w; z) / l_t(w; X)
        - If l_t(w; X) <= e, then a1(w) = 0, a2(w) = 1
        """
        if current_grad.numel() == 0 or ref_grad.numel() == 0:
            return current_grad

        if current_loss > self.epsilon:
            # Case 1: Current loss > e
            alpha1 = 1.0
            alpha2 = ref_loss / current_loss if current_loss > 0 else 0.0
        else:
            # Case 2: Current loss <= e  
            alpha1 = 0.0
            alpha2 = 1.0

        # Balance the gradients: a1 * current_grad + a2 * ref_grad
        balanced_grad = alpha1 * current_grad + alpha2 * ref_grad
        
        return balanced_grad

    def apply_si_regularization_to_gradient(self, grad_vector):
        """Apply SI regularization directly to gradient vector"""
        if grad_vector.numel() == 0:
            return grad_vector
            
        # Compute SI gradient contribution
        si_grad_components = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.si_omega:
                # SI gradient: lambda * 2 * omega * (theta - theta_prev)
                param_diff = param - self.si_prev_params[name]
                si_grad = 2 * self.si_lambda * self.si_omega[name] * param_diff
                si_grad_components.append(si_grad.view(-1))
        
        if si_grad_components:
            si_grad_vector = torch.cat(si_grad_components)
            return grad_vector + si_grad_vector
        else:
            return grad_vector
        
    def set_gradient(self, grad_vector):
        """Set model gradients from flattened gradient vector"""
        if grad_vector.numel() == 0:
            return

        pointer = 0
        for param in self.model.parameters():
            if param.grad is not None:
                num_param = param.numel()
                param.grad.data = grad_vector[pointer : pointer + num_param].view(
                    param.shape
                )
                pointer += num_param

    def optimize(self, data, labels, memory_samples=None):
        """MEGA-I optimization with Synaptic Intelligence and loss-based gradient balancing"""
        # Move data to device
        data, labels = data.to(self.device), labels.to(self.device)
        
        # Compute base loss for tracking (without SI regularization for MEGA-I)
        self.model.zero_grad()
        outputs = self.model(data)
        base_loss = self.criterion(outputs, labels)
        current_loss = base_loss.item()

        # Update learning rate if scheduler is provided
        if self.lr_scheduler is not None:
            new_lr = self.lr_scheduler.step(current_loss)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

        # Compute gradient on current task (WITHOUT SI regularization for pure task gradient)
        base_loss.backward()
        current_grad = []
        for param in self.model.parameters():
            if param.grad is not None:
                current_grad.append(param.grad.view(-1))
        current_grad = torch.cat(current_grad) if current_grad else torch.tensor([]).to(self.device)

        # If we have memory samples, apply MEGA-I gradient balancing
        if memory_samples is not None and len(memory_samples) > 0:
            # Prepare memory data
            mem_data_list = []
            mem_labels_list = []

            # Handle memory samples correctly
            try:
                for sample_item in memory_samples[: self.eps_mem_batch]:
                    if isinstance(sample_item, (list, tuple)) and len(sample_item) >= 2:
                        sample_data, sample_label = sample_item[0], sample_item[1]
                    else:
                        # Assume it's just data if not tuple/list
                        sample_data, sample_label = sample_item, 0

                    mem_data_list.append(sample_data)
                    mem_labels_list.append(sample_label)

                if mem_data_list:
                    mem_data = torch.stack(mem_data_list).to(self.device)
                    mem_labels = (
                        torch.stack(mem_labels_list).to(self.device)
                        if isinstance(mem_labels_list[0], torch.Tensor)
                        else torch.tensor(mem_labels_list).to(self.device)
                    )

                    # Compute reference gradient on memory (WITHOUT SI regularization for pure task gradient)
                    ref_grad, ref_loss = self.compute_gradient(mem_data, mem_labels, include_si_reg=False)

                    # Apply MEGA-I gradient balancing (on pure task gradients)
                    balanced_grad = self.mega_i_gradient_balance(
                        current_grad, ref_grad, current_loss, ref_loss
                    )

                    # Now apply SI regularization to the balanced gradient
                    final_grad = self.apply_si_regularization_to_gradient(balanced_grad)

                    # Set final gradient and optimize
                    self.set_gradient(final_grad)
                    self.optimizer.step()
                else:
                    # No valid memory samples, apply SI regularization to current gradient
                    final_grad = self.apply_si_regularization_to_gradient(current_grad)
                    self.set_gradient(final_grad)
                    self.optimizer.step()
            except Exception as e:
                # Fallback to regular update with SI regularization
                print(f"Memory processing failed: {e}")
                final_grad = self.apply_si_regularization_to_gradient(current_grad)
                self.set_gradient(final_grad)
                self.optimizer.step()
        else:
            # No memory samples, apply SI regularization to current gradient
            final_grad = self.apply_si_regularization_to_gradient(current_grad)
            self.set_gradient(final_grad)
            self.optimizer.step()

        # Update Synaptic Intelligence tracking
        self.update_si_tracking(current_loss)
        
        # Automatic importance consolidation (truly task-agnostic)
        if self.si_update_freq > 0:
            self.si_step_count += 1
            if self.si_step_count % self.si_update_freq == 0:
                self.consolidate_si_importance(reset_accumulation=False)  # Don't reset for continuous learning

        return current_loss

    # def task_finished(self):
    #     """
    #     OPTIONAL: Call this method when a task is completed to consolidate SI importance weights.
    #     This is provided for compatibility but the system works task-agnostically without it.
    #     """
    #     self.consolidate_si_importance(reset_accumulation=True)
    #     print("Task completed - Synaptic Intelligence importance weights consolidated.")

    # def get_si_importance_stats(self):
    #     """Get statistics about current SI importance weights for analysis"""
    #     stats = {}
    #     for name, omega in self.si_omega.items():
    #         stats[name] = {
    #             'mean': omega.mean().item(),
    #             'std': omega.std().item(),
    #             'max': omega.max().item(),
    #             'min': omega.min().item(),
    #             'nonzero_ratio': (omega > 0).float().mean().item()
    #         }
    #     return stats


def evaluate_all_tasks(model, criterion, task_dataloaders, device):
    """Evaluate model on all tasks using test data and return average accuracy"""
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for task_dataloader in task_dataloaders:
            for data, labels in task_dataloader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

    return total_correct / total_samples if total_samples > 0 else 0.0


def evaluate_tasks_up_to(model, criterion, task_dataloaders, current_task_id, device):
    """Evaluate model only on tasks seen so far using test data"""
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for task_id in range(current_task_id + 1):
            task_dataloader = task_dataloaders[task_id]
            for data, labels in task_dataloader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

    return total_correct / total_samples if total_samples > 0 else 0.0


def evaluate_single_task(model, criterion, task_dataloader, device):
    """Evaluate model on a single task using test data and return accuracy"""
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, labels in task_dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    return total_correct / total_samples if total_samples > 0 else 0.0
