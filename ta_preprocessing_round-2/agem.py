import torch
import torch.nn.functional as F

class AGEMHandler:
    def __init__(self, model, criterion, optimizer, device, batch_size=256):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.eps_mem_batch = batch_size  # Memory batch size for gradient computation
        self.device = device

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
                projected_grad = current_grad - (dot_product / ref_grad_norm_sq) * ref_grad
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
                param.grad.data = grad_vector[pointer:pointer + num_param].view(param.shape)
                pointer += num_param

    def compute_and_project_batch_gradient(self, data, labels, memory_samples=None):
        """
        Computes the gradient for a single batch and projects it using memory samples.
        Returns the projected gradient (flattened) and the loss for the batch.
        DOES NOT call optimizer.step() or modify model.grad directly.
        """
        # Compute current task gradient and loss
        # Use a temporary model for compute_gradient to avoid conflicts with global model.grad if possible,
        # but given `compute_gradient`'s save/restore logic, it should be fine.
        current_grad_flat, batch_loss = self.compute_gradient(data, labels)

        # If we have memory samples, compute reference gradient and project
        if memory_samples is not None and len(memory_samples) > 0:
            mem_data_list = []
            mem_labels_list = []

            try:
                # Use min to avoid index out of bounds if memory is smaller than eps_mem_batch
                for sample_data, sample_label in memory_samples[:self.eps_mem_batch]:
                    mem_data_list.append(sample_data)
                    mem_labels_list.append(sample_label)

                if mem_data_list:
                    mem_data = torch.stack(mem_data_list).to(self.device)
                    # Handle labels which could be tensors or integers
                    mem_labels = torch.stack(mem_labels_list).to(self.device) if isinstance(mem_labels_list[0],
                                                                                            torch.Tensor) else torch.tensor(
                        mem_labels_list).to(self.device)

                    ref_grad_flat, _ = self.compute_gradient(mem_data, mem_labels)  # We only need the gradient here

                    projected_grad_flat = self.project_gradient(current_grad_flat, ref_grad_flat)
                    return projected_grad_flat, batch_loss
                else:
                    # No valid memory samples after filtering, return original gradient
                    return current_grad_flat, batch_loss
            except Exception as e:
                print(f"Memory processing failed during gradient projection: {e}")
                # If an error occurs with memory, proceed without projection
                return current_grad_flat, batch_loss
        else:
            # No memory samples, return original gradient
            return current_grad_flat, batch_loss

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

        pointer = 0
        for param in self.model.parameters():
            num_param = param.numel()
            if pointer + num_param > accumulated_projected_grad_vector.numel():
                print(
                    f"Error: Gradient vector size mismatch for parameter. Expected {num_param}, available {accumulated_projected_grad_vector.numel() - pointer}")
                # Handle gracefully, e.g., break or raise error
                break

            # Reshape and assign the accumulated gradient
            param.grad = accumulated_projected_grad_vector[pointer:pointer + num_param].view(param.shape)
            pointer += num_param

        self.optimizer.step()

    def optimize(self, data, labels, memory_samples=None):
        """A-GEM optimization with gradient projection"""
        # Compute loss for tracking
        self.model.zero_grad()
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        current_loss = loss.item()

        # Compute gradient on current task
        loss.backward()
        current_grad = []
        for param in self.model.parameters():
            if param.grad is not None:
                current_grad.append(param.grad.view(-1))
        current_grad = torch.cat(current_grad) if current_grad else torch.tensor([])

        # If we have memory samples, compute reference gradient and project
        if memory_samples is not None and len(memory_samples) > 0:
            # Prepare memory data
            mem_data_list = []
            mem_labels_list = []

            # Handle memory samples correctly
            try:
                for sample_item in memory_samples[:self.eps_mem_batch]:
                    if isinstance(sample_item, (list, tuple)) and len(sample_item) >= 2:
                        sample_data, sample_label = sample_item[0], sample_item[1]
                    else:
                        # Assume it's just data if not tuple/list
                        sample_data, sample_label = sample_item, 0

                    mem_data_list.append(sample_data)
                    mem_labels_list.append(sample_label)

                if mem_data_list:
                    mem_data = torch.stack(mem_data_list).to(self.device)
                    mem_labels = torch.stack(mem_labels_list).to(self.device) if isinstance(mem_labels_list[0], torch.Tensor) else torch.tensor(mem_labels_list).to(self.device)

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
