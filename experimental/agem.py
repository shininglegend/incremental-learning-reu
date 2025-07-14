import torch
import torch.nn.functional as F


class AGEMHandler:
    def __init__(self, model, criterion, optimizer, device, batch_size=256):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.eps_mem_batch = batch_size
        self.device = device

    def extract_gradients(self):
        """Extract flattened gradients from model parameters"""
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        return torch.cat(grads) if grads else torch.tensor([], device=self.device)

    def set_gradients(self, grad_vector):
        """Set model gradients from flattened gradient vector"""
        if grad_vector.numel() == 0:
            return

        pointer = 0
        for param in self.model.parameters():
            if param.grad is not None:
                num_param = param.numel()
                param.grad.data.copy_(grad_vector[pointer:pointer + num_param].view(param.shape))
                pointer += num_param

    def project_gradient(self, current_grad, ref_grad):
        """Project current gradient to not increase loss on reference gradient"""
        if ref_grad.numel() == 0 or current_grad.numel() == 0:
            return current_grad

        dot_product = torch.dot(current_grad, ref_grad)

        if dot_product < 0:
            ref_grad_norm_sq = torch.dot(ref_grad, ref_grad)
            if ref_grad_norm_sq > 0:
                return current_grad - (dot_product / ref_grad_norm_sq) * ref_grad

        return current_grad

    def compute_and_project_batch_gradient(self, data, labels, memory_samples=None):
        """
        Computes the gradient for a single batch and projects it using memory samples.
        Returns the projected gradient (flattened) and the loss for the batch.
        DOES NOT call optimizer.step() or modify model.grad directly.
        """
        # Move data to device
        data, labels = data.to(self.device), labels.to(self.device)

        # Forward pass and compute loss
        self.model.zero_grad()
        outputs = self.model(data)
        batch_loss = self.criterion(outputs, labels)

        # Backward pass to get current gradient
        batch_loss.backward()
        current_grad = self.extract_gradients()

        # Project gradient if we have memory samples
        if memory_samples is not None and len(memory_samples) > 0:
            projected_grad = self._project_with_memory(current_grad, memory_samples)
            if projected_grad is not None:
                return projected_grad, batch_loss.item()

        # Return original gradient if no projection needed
        return current_grad, batch_loss.item()

    def optimize(self, data, labels, memory_samples=None):
        """Optimized A-GEM optimization with gradient projection"""

        # Move data to device
        data, labels = data.to(self.device), labels.to(self.device)

        # Forward pass and loss computation
        self.model.zero_grad()
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        current_loss = loss.item()

        # Backward pass to compute current gradients
        loss.backward()
        current_grad = self.extract_gradients()

        # Project gradient if we have memory samples
        if memory_samples is not None and len(memory_samples) > 0:
            projected_grad = self._project_with_memory(current_grad, memory_samples)
            if projected_grad is not None:
                self.set_gradients(projected_grad)

        # Optimize
        self.optimizer.step()
        return current_loss

    def apply_accumulated_gradients(self, accumulated_projected_grad_vector):
        """
        Sets the model's gradients from an accumulated (and projected) gradient vector
        and then calls optimizer.step().
        This method should be called ONCE per update step (e.g., per epoch or accumulation steps).
        """
        if accumulated_projected_grad_vector.numel() == 0:
            print("Warning: No accumulated gradients to apply.")
            return

        self.model.zero_grad()  # Clear any existing gradients before setting new ones

        # Set the accumulated gradients
        self.set_gradients(accumulated_projected_grad_vector)

        # Apply the optimizer step
        self.optimizer.step()

    def _project_with_memory(self, current_grad, memory_samples):
        """Helper method to project gradient using memory samples"""
        try:
            # Batch process memory samples
            mem_data_list = []
            mem_labels_list = []

            # Take only what we need
            samples_to_use = memory_samples[:self.eps_mem_batch]

            for sample_item in samples_to_use:
                if isinstance(sample_item, (list, tuple)) and len(sample_item) >= 2:
                    sample_data, sample_label = sample_item[0], sample_item[1]
                else:
                    continue  # Skip invalid samples

                mem_data_list.append(sample_data)
                mem_labels_list.append(sample_label)

            if not mem_data_list:
                return None

            # Stack memory data efficiently
            mem_data = torch.stack(mem_data_list).to(self.device)

            # Handle labels properly
            if isinstance(mem_labels_list[0], torch.Tensor):
                if mem_labels_list[0].numel() == 1:
                    mem_labels = torch.stack(mem_labels_list).to(self.device)
                else:
                    mem_labels = torch.cat([label.flatten() for label in mem_labels_list]).to(self.device)
            else:
                mem_labels = torch.tensor(mem_labels_list, device=self.device)

            # Compute reference gradient efficiently
            ref_grad = self._compute_reference_gradient(mem_data, mem_labels)

            # Project gradient
            return self.project_gradient(current_grad, ref_grad)

        except Exception as e:
            print(f"Memory processing failed: {e}")
            return None

    def _compute_reference_gradient(self, mem_data, mem_labels):
        """Compute reference gradient efficiently without saving/restoring"""
        # Store current gradients temporarily
        current_grads = [param.grad.clone() if param.grad is not None else None
                         for param in self.model.parameters()]

        # Compute reference gradient
        self.model.zero_grad()
        mem_outputs = self.model(mem_data)
        mem_loss = self.criterion(mem_outputs, mem_labels)
        mem_loss.backward()

        # Extract reference gradient
        ref_grad = self.extract_gradients()

        # Restore original gradients efficiently
        for param, original_grad in zip(self.model.parameters(), current_grads):
            param.grad = original_grad

        return ref_grad


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
