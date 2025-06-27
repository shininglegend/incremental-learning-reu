import torch
import torch.nn.functional as F

class AGEMHandler:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.eps_mem_batch = 256  # Memory batch size for gradient computation

    def compute_gradient(self, data, labels):
        """Compute gradients for given data and labels"""
        self.model.zero_grad()
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        loss.backward()

        # Extract gradients
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        return torch.cat(grads) if grads else torch.tensor([])

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

    def optimize(self, data, labels, memory_samples):
        """A-GEM optimization with gradient projection"""
        # Compute gradient on current task
        current_grad = self.compute_gradient(data, labels)

        # If we have memory samples, compute reference gradient and project
        if memory_samples and len(memory_samples) > 0:
            # Prepare memory data
            mem_data_list = []
            mem_labels_list = []

            for sample_data, sample_label in memory_samples[:self.eps_mem_batch]:
                mem_data_list.append(sample_data)
                mem_labels_list.append(sample_label)

            if mem_data_list:
                mem_data = torch.stack(mem_data_list)
                mem_labels = torch.stack(mem_labels_list) if isinstance(mem_labels_list[0], torch.Tensor) else torch.tensor(mem_labels_list)

                # Compute reference gradient on memory
                ref_grad = self.compute_gradient(mem_data, mem_labels)

                # Project current gradient
                projected_grad = self.project_gradient(current_grad, ref_grad)

                # Set projected gradient and optimize
                self.model.zero_grad()
                self.set_gradient(projected_grad)
                self.optimizer.step()
            else:
                # No valid memory samples, just do regular update
                self.optimizer.step()
        else:
            # No memory samples, just do regular update
            self.optimizer.step()

def evaluate_all_tasks(model, criterion, task_dataloaders):
    """Evaluate model on all tasks and return average accuracy"""
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for task_dataloader in task_dataloaders:
            for data, labels in task_dataloader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

    return total_correct / total_samples if total_samples > 0 else 0.0
