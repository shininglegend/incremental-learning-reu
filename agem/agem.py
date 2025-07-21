import torch
import torch.nn.functional as F


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
        si_c=0.1,  # SI regularization strength
        xi=1.0,  # SI damping parameter
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.eps_mem_batch = batch_size  # Memory batch size for gradient computation
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.epsilon = epsilon  # Sensitivity parameter for MEGA-I
        
        # Synaptic Intelligence parameters
        self.si_c = si_c  # Regularization strength
        self.xi = xi  # Damping parameter
        
        # Initialize SI variables
        self._initialize_si()

    def _initialize_si(self):
        """Initialize Synaptic Intelligence tracking variables"""
        self.si_omega = {}  # Importance weights for each parameter
        self.si_w_old = {}  # Previous parameter values
        self.si_delta_w = {}  # Parameter changes during current task
        self.si_gradients = {}  # Accumulated gradients during current task
        
        # Initialize for each parameter
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.si_omega[name] = torch.zeros_like(param.data).to(self.device)
                self.si_w_old[name] = param.data.clone().to(self.device)
                self.si_delta_w[name] = torch.zeros_like(param.data).to(self.device)
                self.si_gradients[name] = torch.zeros_like(param.data).to(self.device)

    def update_si_gradients(self):
        """Update accumulated gradients for SI importance calculation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and name in self.si_gradients:
                self.si_gradients[name] += param.grad.data

    def update_si_deltas(self):
        """Update parameter changes for SI importance calculation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.si_delta_w:
                self.si_delta_w[name] += param.data - self.si_w_old[name]
                self.si_w_old[name] = param.data.clone()

    def compute_si_importance(self):
        """Compute importance weights using Synaptic Intelligence"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.si_omega:
                # Compute importance: |gradient * delta_w| / (delta_w^2 + xi)
                delta_w = self.si_delta_w[name]
                grad_sum = self.si_gradients[name]
                
                # Importance calculation with damping
                importance = torch.abs(grad_sum * delta_w) / (delta_w.pow(2) + self.xi)
                
                # Update omega (importance weights)
                self.si_omega[name] += importance
                
                # Reset accumulators for next task
                self.si_delta_w[name].zero_()
                self.si_gradients[name].zero_()

    def compute_si_loss(self):
        """Compute Synaptic Intelligence regularization loss"""
        si_loss = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.si_omega:
                # SI penalty: c/2 * sum(omega * (w - w_old)^2)
                w_old = self.si_w_old[name]
                omega = self.si_omega[name]
                si_loss += (omega * (param - w_old).pow(2)).sum()
        
        return self.si_c / 2 * si_loss

    def compute_gradient(self, data, labels, include_si=True):
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
        
        # Add SI regularization loss if requested
        if include_si:
            si_loss = self.compute_si_loss()
            loss += si_loss

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

        return torch.cat(grads) if grads else torch.tensor([]).to(self.device), loss.item()

    def compute_loss(self, data, labels, include_si=True):
        """Compute loss for given data and labels without affecting gradients"""
        data, labels = data.to(self.device), labels.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            
            # Add SI regularization loss if requested
            if include_si:
                si_loss = self.compute_si_loss()
                loss += si_loss
        
        return loss.item()

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
        """MEGA-I optimization with loss-based gradient balancing and Synaptic Intelligence"""
        # Move data to device
        data, labels = data.to(self.device), labels.to(self.device)
        
        # Compute loss for tracking (including SI regularization)
        self.model.zero_grad()
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        
        # Add SI regularization loss
        si_loss = self.compute_si_loss()
        total_loss = loss + si_loss
        current_loss = total_loss.item()

        # Update learning rate if scheduler is provided
        if self.lr_scheduler is not None:
            new_lr = self.lr_scheduler.step(current_loss)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

        # Compute gradient on current task (including SI)
        total_loss.backward()
        
        # Update SI gradient accumulation
        self.update_si_gradients()
        
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

                    # Compute reference gradient and loss on memory (including SI)
                    ref_grad, ref_loss = self.compute_gradient(mem_data, mem_labels, include_si=True)

                    # Apply MEGA-I gradient balancing
                    balanced_grad = self.mega_i_gradient_balance(
                        current_grad, ref_grad, current_loss, ref_loss
                    )

                    # Set balanced gradient and optimize
                    self.set_gradient(balanced_grad)
                    self.optimizer.step()
                    
                    # Update SI parameter deltas after optimization step
                    self.update_si_deltas()
                else:
                    # No valid memory samples, just do regular update
                    self.optimizer.step()
                    self.update_si_deltas()
            except Exception as e:
                # Fallback to regular update if memory processing fails
                print(f"Memory processing failed: {e}")
                self.optimizer.step()
                self.update_si_deltas()
        else:
            # No memory samples, just do regular update
            self.optimizer.step()
            self.update_si_deltas()

        return current_loss

    def end_task(self):
        """Call this method when finishing a task to update SI importance weights"""
        print("Computing Synaptic Intelligence importance weights...")
        self.compute_si_importance()
        print("SI importance weights updated.")

    def get_si_stats(self):
        """Get statistics about SI importance weights for monitoring"""
        stats = {}
        for name, omega in self.si_omega.items():
            stats[name] = {
                'mean_importance': omega.mean().item(),
                'max_importance': omega.max().item(),
                'std_importance': omega.std().item(),
                'nonzero_params': (omega > 0).sum().item(),
                'total_params': omega.numel()
            }
        return stats

    def save_si_state(self, filepath):
        """Save SI state for checkpointing"""
        si_state = {
            'si_omega': {name: omega.cpu() for name, omega in self.si_omega.items()},
            'si_w_old': {name: w.cpu() for name, w in self.si_w_old.items()},
            'si_c': self.si_c,
            'xi': self.xi
        }
        torch.save(si_state, filepath)

    def load_si_state(self, filepath):
        """Load SI state from checkpoint"""
        si_state = torch.load(filepath, map_location=self.device)
        self.si_omega = {name: omega.to(self.device) for name, omega in si_state['si_omega'].items()}
        self.si_w_old = {name: w.to(self.device) for name, w in si_state['si_w_old'].items()}
        self.si_c = si_state['si_c']
        self.xi = si_state['xi']
        
        # Reinitialize accumulators
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.si_omega:
                self.si_delta_w[name] = torch.zeros_like(param.data).to(self.device)
                self.si_gradients[name] = torch.zeros_like(param.data).to(self.device)


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
