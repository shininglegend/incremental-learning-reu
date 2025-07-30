import torch
import numpy as np

class AGEMHandler:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion,
        optimizer,
        device,
        batch_size=256,
        lr_scheduler=None,

        # MAS parameters
        mas_weight=0.5,  # MAS regularization weight
        loss_window_length=5,  # Window size for loss plateau detection
        loss_window_mean_threshold=0.2,  # Mean threshold for plateau detection
        loss_window_variance_threshold=0.1,  # Variance threshold for plateau detection
        importance_update_frequency=100,  # Steps between importance weight updates
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.eps_mem_batch = batch_size  # Memory batch size for gradient computation
        self.device = device
        self.lr_scheduler = lr_scheduler


        # MAS parameters
        self.mas_weight = mas_weight
        self.loss_window_length = loss_window_length
        self.loss_window_mean_threshold = loss_window_mean_threshold
        self.loss_window_variance_threshold = loss_window_variance_threshold
        self.importance_update_frequency = importance_update_frequency
        
        # MAS state variables
        self.star_variables = []  # Anchor parameters
        self.omegas = []  # Importance weights
        self.loss_window = []  # Loss window for plateau detection
        self.update_count = 0  # Number of importance weight updates
        self.step_count = 0  # Total optimization steps
        self.new_peak_detected = True  # Flag for peak detection
        self.last_loss_window_mean = float('inf')
        self.last_loss_window_variance = float('inf')
        
        # Initialize importance weights to zero
        self._initialize_importance_weights()

    def _initialize_importance_weights(self):
        """Initialize importance weights and anchor parameters"""
        self.omegas = []
        self.star_variables = []
        
        for param in self.model.parameters():
            # Initialize importance weights to zero
            self.omegas.append(torch.zeros_like(param.data))
            # Initialize anchor parameters
            self.star_variables.append(param.data.clone().detach())

    
    def _detect_loss_plateau(self, current_loss):
        """
        Detect loss plateaus using loss window statistics
        Returns True if importance weights should be updated
        """
        # Add loss to window
        self.loss_window.append(current_loss)
        if len(self.loss_window) > self.loss_window_length:
            self.loss_window.pop(0)
        
        if len(self.loss_window) < self.loss_window_length:
            return False
            
        # Calculate window statistics
        loss_window_mean = np.mean(self.loss_window)
        loss_window_variance = np.var(self.loss_window)
        
        # Check for new peak
        if not self.new_peak_detected and loss_window_mean > self.last_loss_window_mean + np.sqrt(self.last_loss_window_variance):
            self.new_peak_detected = True
        
        # Check for plateau conditions
        plateau_detected = (
            loss_window_mean < self.loss_window_mean_threshold and
            loss_window_variance < self.loss_window_variance_threshold and
            self.new_peak_detected
        )
        
        if plateau_detected:
            self.last_loss_window_mean = loss_window_mean
            self.last_loss_window_variance = loss_window_variance
            self.new_peak_detected = False
            return True
            
        return False

    def _update_importance_weights(self, memory_samples=None):
        """
        Update MAS importance weights based on gradient magnitudes
        Uses memory samples if available, otherwise uses current model state
        """
        self.update_count += 1
        
        # Calculate gradients for importance estimation
        gradients = [torch.zeros_like(param) for param in self.model.parameters()]
        
        if memory_samples is not None and len(memory_samples) > 0:
            # Use memory samples for importance calculation
            try:
                for sample_item in memory_samples[:min(len(memory_samples), 50)]:  # Limit samples for efficiency
                    if isinstance(sample_item, (list, tuple)) and len(sample_item) >= 2:
                        sample_data, sample_label = sample_item[0], sample_item[1]
                    else:
                        sample_data, sample_label = sample_item, 0
                    
                    # Move to device
                    if isinstance(sample_data, torch.Tensor):
                        sample_data = sample_data.unsqueeze(0).to(self.device)
                    if isinstance(sample_label, torch.Tensor):
                        sample_label = sample_label.unsqueeze(0).to(self.device)
                    else:
                        sample_label = torch.tensor([sample_label]).to(self.device)
                    
                    # Compute gradients for this sample
                    self.model.zero_grad()
                    output = self.model(sample_data)
                    
                    # Use L2 norm of output for importance (as in original MAS)
                    importance_loss = torch.norm(output, 2, dim=1).sum()
                    importance_loss.backward()
                    
                    # Accumulate absolute gradients
                    for i, param in enumerate(self.model.parameters()):
                        if param.grad is not None:
                            gradients[i] += torch.abs(param.grad.data)
                            
            except Exception as e:
                print(f"Error in importance weight calculation with memory: {e}")
                # Fallback to using current parameters
                gradients = [torch.zeros_like(param) for param in self.model.parameters()]
        
        # Update running average of importance weights
        old_omegas = [omega.clone() for omega in self.omegas]
        
        for i, (param, gradient) in enumerate(zip(self.model.parameters(), gradients)):
            if self.update_count == 1:
                # First update: use gradient directly
                self.omegas[i] = gradient.clone()
            else:
                # Running average: omega = (1/n) * new_gradient + (1 - 1/n) * old_omega
                alpha = 1.0 / self.update_count
                self.omegas[i] = alpha * gradient + (1 - alpha) * old_omegas[i]
            
            # Update anchor parameters
            self.star_variables[i] = param.data.clone().detach()

    def _compute_mas_regularization(self):
        """Compute MAS regularization term"""
        mas_loss = 0.0
        
        if len(self.star_variables) > 0 and len(self.omegas) > 0:
            for param, star_param, omega in zip(self.model.parameters(), self.star_variables, self.omegas):
                # MAS regularization: (lambda/2) * sum(omega * (theta - theta_star)^2)
                mas_loss += torch.sum(omega * (param - star_param) ** 2)
            
            mas_loss = (self.mas_weight / 2.0) * mas_loss
        
        return mas_loss
    
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
        #loss.backward() # Keep commented out because backward is called right below

        # Add MAS regularization
        mas_loss = self._compute_mas_regularization()
        total_loss = loss + mas_loss
        
        total_loss.backward()

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

    def project_gradient(self, current_grad, ref_grad):
        """Project current gradient to not increase loss on reference gradient"""
        if ref_grad.numel() == 0 or current_grad.numel() == 0:
            return current_grad

        # Compute dot product
        dot_product = torch.dot(current_grad, ref_grad)

        # If dot product is negative, project the gradient
        if dot_product < 0:
            ref_grad_norm_sq = torch.dot(ref_grad, ref_grad)
            if ref_grad_norm_sq > 0:
                projected_grad = (
                    current_grad - (dot_product / ref_grad_norm_sq) * ref_grad
                )
                return projected_grad

        return current_grad

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
        """A-GEM optimization with gradient projection and MAS regularization and TA importance weight updates"""
        
        self.step_count += 1
        
        # Compute loss for tracking (without MAS regularization for plateau detection)
        self.model.zero_grad()
        outputs = self.model(data)
        base_loss = self.criterion(outputs, labels)
        current_loss = base_loss.item()
        
        # Add MAS regularization to the loss
        mas_loss = self._compute_mas_regularization()
        loss = base_loss + mas_loss

        # Update learning rate if scheduler is provided
        if self.lr_scheduler is not None:
            new_lr = self.lr_scheduler.step(current_loss)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

        # Compute gradient on current task (with MAS regularization)
        loss.backward()
        current_grad = []
        for param in self.model.parameters():
            if param.grad is not None:
                current_grad.append(param.grad.view(-1))
        current_grad = torch.cat(current_grad) if current_grad else torch.tensor([])

        # Check for loss plateau and update importance weights if needed
        if self._detect_loss_plateau(current_loss):
            #print(f"Plateau detected at step {self.step_count}, updating importance weights...")
            self._update_importance_weights(memory_samples)

        # Periodic importance weight updates (alternative to plateau detection)
        elif self.step_count % self.importance_update_frequency == 0:
            #print(f"Periodic importance update at step {self.step_count}")
            self._update_importance_weights(memory_samples)

        # If we have memory samples, compute reference gradient and project
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

                    # Compute reference gradient on memory
                    ref_grad = self.compute_gradient(mem_data, mem_labels)

                    # Project current gradient
                    projected_grad = self.project_gradient(current_grad, ref_grad)

                    # Set projected gradient and optimize
                    self.set_gradient(projected_grad)
                    self.optimizer.step()
                else:
                    # No valid memory samples, just do regular update
                    self.optimizer.step()
            except Exception as e:
                # Fallback to regular update if memory processing fails
                print(f"Memory processing failed: {e}")
                self.optimizer.step()
        else:
            # No memory samples, just do regular update
            self.optimizer.step()

        return current_loss

    def get_importance_stats(self):
        """Get statistics about current importance weights"""
        if not self.omegas:
            return {"mean": 0, "std": 0, "max": 0, "min": 0, "updates": self.update_count}
        
        all_omegas = torch.cat([omega.flatten() for omega in self.omegas])
        return {
            "mean": all_omegas.mean().item(),
            "std": all_omegas.std().item(),
            "max": all_omegas.max().item(),
            "min": all_omegas.min().item(),
            "updates": self.update_count
        }