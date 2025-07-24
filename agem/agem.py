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
        epsilon=0.00001,  # Now meaningful for normalized losses (0-1 range)
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.eps_mem_batch = batch_size
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.epsilon = epsilon

    def start_epoch(self):
        """Call this at the beginning of each epoch"""
        self.epoch_min_loss = float('inf')
        self.epoch_max_loss = float('-inf')

    def normalize_loss_sliding_window(self, loss_value, add_to_history=True):
        """Normalize using a sliding window of recent losses"""
        if not hasattr(self, 'loss_history'):
            self.loss_history = []
        
        # Only add to history if requested (for current losses, not reference losses)
        if add_to_history:
            self.loss_history.append(loss_value)
            
            # Keep only last N losses for normalization bounds
            window_size = 200  # Adjust as needed (50-250, ...)
            if len(self.loss_history) > window_size:
                self.loss_history = self.loss_history[-window_size:]
        
        if len(self.loss_history) < 2:
            return 0.5  # Default for insufficient data
        
        # For consistent normalization, always include the current value in bounds calculation
        # but only permanently store it if add_to_history=True
        all_losses = self.loss_history if add_to_history else self.loss_history + [loss_value]
        min_val = min(all_losses)
        max_val = max(all_losses)

        if min_val == max_val:
            return 0.5

        return (loss_value - min_val) / (max_val - min_val)

    def normalize_reference_loss(self, ref_loss):
        """
        Normalize reference loss using the same approach as current loss
        but without adding to history permanently.
        """
        return self.normalize_loss_sliding_window(ref_loss, add_to_history=False)

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
        normalized_current_loss = self.normalize_loss_sliding_window(current_loss, add_to_history=True)
        # Normalize the reference loss
        normalized_ref_loss = self.normalize_reference_loss(ref_loss)
        
        #print(f"Original loss: {current_loss:.6f}, Normalized loss: {normalized_current_loss:.6f}, epsilon: {self.epsilon}")

        if normalized_current_loss > self.epsilon:
            # Case 1: Normalized current loss > epsilon
            #print("| ========== | CASE 1111111111 | ========== |")
            #print("*****************************************************************************************")
            alpha1 = 1.0
            alpha2 = normalized_ref_loss / normalized_current_loss if normalized_current_loss > 0 else 0.0
        else:
            # Case 2: Normalized current loss <= epsilon
            #print("| ========== | CASE 2222222222 | ========== |")
            #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
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
                    #print("balanced grad: ", balanced_grad)

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
