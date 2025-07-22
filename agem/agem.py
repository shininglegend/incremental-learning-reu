import torch


class AGEMHandler:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion,
        optimizer,
        device,
        batch_size=256,
        lr_scheduler=None,
        epsilon=1e-32,  # Now meaningful for normalized losses (0-1 range)
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.eps_mem_batch = batch_size
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.epsilon = epsilon
        
        # For running min-max normalization
        self.epoch_min_loss = float('inf')
        self.epoch_max_loss = float('-inf')

    def start_epoch(self):
        """Call this at the beginning of each epoch"""
        self.epoch_min_loss = float('inf')
        self.epoch_max_loss = float('-inf')

    def update_loss_bounds(self, loss_value):
        """Update running min and max loss values"""
        self.epoch_min_loss = min(self.epoch_min_loss, loss_value)
        self.epoch_max_loss = max(self.epoch_max_loss, loss_value)

    # Added, could do without (along with small changes to function below it)
    def min_max_normalize(self, data):
        """
        Normalize a list of numbers using Min-Max scaling to the range [0, 1].
        Parameters:
            data (list or array-like): A list of numerical values.
        Returns:
            list: Normalized values in the range [0, 1].
        """
        min_val = min(data)
        max_val = max(data)
        
        if min_val == max_val:
            # Avoid division by zero if all values are the same
            return [0.0 for _ in data]
        
        return [(x - min_val) / (max_val - min_val) for x in data]

    def normalize_loss(self, loss_value):
        """Wrapper to use the min_max_normalize function for a single loss value"""
        # For single loss normalization, we need to use the epoch bounds
        if self.epoch_max_loss == self.epoch_min_loss:
            return 0.5  # Return middle value if all losses are the same
        
        if self.epoch_min_loss == float('inf') or self.epoch_max_loss == float('-inf'):
            return 0.5  # Return middle value if bounds not initialized
        
        # Use the min_max_normalize approach
        data = [self.epoch_min_loss, self.epoch_max_loss, loss_value]
        normalized_data = self.min_max_normalize(data)
        return normalized_data[2]  # Return the normalized version of loss_value

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

        return torch.cat(grads) if grads else torch.tensor([]).to(self.device), loss.item()

    def compute_loss(self, data, labels):
        """Compute loss for given data and labels without affecting gradients"""
        data, labels = data.to(self.device), labels.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
        
        return loss.item()

    def mega_i_gradient_balance(self, current_grad, ref_grad, current_loss, ref_loss):
        """
        Balance current and reference gradients using MEGA-I approach based on loss information.
        
        According to equation:
        - If l_t(w; X) > e, then a1(w) = 1, a2(w) = l_ref(w; z) / l_t(w; X)
        - If l_t(w; X) <= e, then a1(w) = 0, a2(w) = 1

        Normalized loss added
        """
        if current_grad.numel() == 0 or ref_grad.numel() == 0:
            return current_grad

        # Normalize the current loss
        normalized_current_loss = self.normalize_loss(current_loss)
        
        print(f"Original loss: {current_loss:.6f}, Normalized loss: {normalized_current_loss:.6f}, epsilon: {self.epsilon}")

        if normalized_current_loss > self.epsilon:
            # Case 1: Normalized current loss > epsilon
            alpha1 = 1.0
            alpha2 = ref_loss / current_loss if current_loss > 0 else 0.0
        else:
            # Case 2: Normalized current loss <= epsilon
            print("| ========== | IN THE DEFAULT CASE (LOW NORMALIZED LOSS) | ========== |")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            alpha1 = 0.0
            alpha2 = 1.0

        # Balance the gradients
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
        """MEGA-I optimization with normalized loss-based gradient balancing"""
        data, labels = data.to(self.device), labels.to(self.device)
        
        # Compute loss for tracking
        self.model.zero_grad()
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        current_loss = loss.item()

        # Update running min/max bounds
        self.update_loss_bounds(current_loss)

        #Update learning rate if scheduler is provided
        if self.lr_scheduler is not None:
            new_lr = self.lr_scheduler.step(current_loss)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

        # Compute gradient on current task
        loss.backward()
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

                    # Compute reference gradient and loss on memory
                    ref_grad, ref_loss = self.compute_gradient(mem_data, mem_labels)

                    # Apply MEGA-I gradient balancing with normalized loss
                    balanced_grad = self.mega_i_gradient_balance(
                        current_grad, ref_grad, current_loss, ref_loss
                    )

                    # Set balanced gradient and optimize
                    self.set_gradient(balanced_grad)
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
