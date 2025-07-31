import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class AGEMHandler:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion,
        optimizer,
        device,
        batch_size=256,
        lr_scheduler=None,
        # BGD specific parameters
        bgd_alpha=0.5,  # Balance parameter between current and reference gradients
        bgd_temperature=1.0,  # Temperature for gradient similarity computation
        use_bgd=True,  # Whether to use BGD
        gradient_similarity_threshold=0.1,  # Threshold for considering gradients similar
        max_gradient_norm=1.0,  # Maximum gradient norm for clipping
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.eps_mem_batch = batch_size  # Memory batch size for gradient computation
        self.device = device
        self.lr_scheduler = lr_scheduler
        
        # BGD parameters
        self.bgd_alpha = bgd_alpha
        self.bgd_temperature = bgd_temperature
        self.use_bgd = use_bgd
        self.gradient_similarity_threshold = gradient_similarity_threshold
        self.max_gradient_norm = max_gradient_norm
        
        # BGD state
        self.task_gradients = {}  # Store gradients per task
        self.task_counts = defaultdict(int)  # Count samples per task
        self.running_grad_stats = {}  # Running statistics for gradients

    def compute_gradient(self, data, labels, task_id=None):
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

        gradient_vector = torch.cat(grads) if grads else torch.tensor([]).to(self.device)
        
        # Store gradient for BGD if task_id is provided
        if task_id is not None and self.use_bgd:
            self._update_task_gradient(task_id, gradient_vector)
        
        return gradient_vector

    def _update_task_gradient(self, task_id: str, gradient: torch.Tensor):
        """Update running average of gradients for a specific task"""
        if task_id not in self.task_gradients:
            self.task_gradients[task_id] = gradient.clone()
            self.task_counts[task_id] = 1
        else:
            # Running average update
            count = self.task_counts[task_id]
            self.task_gradients[task_id] = (
                self.task_gradients[task_id] * count + gradient
            ) / (count + 1)
            self.task_counts[task_id] += 1

    def compute_gradient_similarity(self, grad1: torch.Tensor, grad2: torch.Tensor) -> float:
        """Compute cosine similarity between two gradients"""
        if grad1.numel() == 0 or grad2.numel() == 0:
            return 0.0
        
        # Normalize gradients
        grad1_norm = F.normalize(grad1.unsqueeze(0), dim=1)
        grad2_norm = F.normalize(grad2.unsqueeze(0), dim=1)
        
        # Compute cosine similarity
        similarity = torch.mm(grad1_norm, grad2_norm.t()).item()
        return similarity

    def balance_gradients_bgd(self, current_grad: torch.Tensor, task_id: str = None) -> torch.Tensor:
        """Apply BGD to balance gradients across tasks"""
        if not self.use_bgd or len(self.task_gradients) <= 1:
            return current_grad
        
        if task_id is None:
            task_id = "current"
        
        # Get all task gradients except current one
        other_task_grads = []
        for tid, grad in self.task_gradients.items():
            if tid != task_id:
                other_task_grads.append(grad)
        
        if not other_task_grads:
            return current_grad
        
        # Compute average gradient from other tasks
        avg_other_grad = torch.stack(other_task_grads).mean(dim=0)
        
        # Compute similarity between current and average other gradients
        similarity = self.compute_gradient_similarity(current_grad, avg_other_grad)
        
        # Apply temperature scaling to similarity
        similarity_scaled = similarity / self.bgd_temperature
        
        # Compute balance weight based on similarity
        if similarity < self.gradient_similarity_threshold:
            # If gradients are dissimilar, balance more aggressively
            balance_weight = self.bgd_alpha * (1.0 + abs(similarity_scaled))
        else:
            # If gradients are similar, use standard balancing
            balance_weight = self.bgd_alpha
        
        # Balance the gradients
        balanced_grad = (
            (1.0 - balance_weight) * current_grad + 
            balance_weight * avg_other_grad
        )
        
        return balanced_grad

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

    def clip_gradient(self, grad_vector: torch.Tensor) -> torch.Tensor:
        """Clip gradient norm if it exceeds maximum"""
        grad_norm = torch.norm(grad_vector)
        if grad_norm > self.max_gradient_norm:
            return grad_vector * (self.max_gradient_norm / grad_norm)
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

    def optimize(self, data, labels, memory_samples=None, task_id=None):
        """A-GEM + BGD optimization with gradient projection and balancing"""
        # Compute loss for tracking
        self.model.zero_grad()
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        current_loss = loss.item()

        # Update learning rate if scheduler is provided
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

        # Apply BGD balancing
        if self.use_bgd and current_grad.numel() > 0:
            current_grad = self.balance_gradients_bgd(current_grad, task_id)
            
            # Update task gradient statistics
            if task_id is not None:
                self._update_task_gradient(task_id, current_grad)

        # Apply A-GEM if we have memory samples
        if memory_samples is not None and len(memory_samples) > 0:
            try:
                # Prepare memory data
                mem_data_list = []
                mem_labels_list = []

                # Handle memory samples correctly
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

                    # Project current gradient using A-GEM
                    projected_grad = self.project_gradient(current_grad, ref_grad)
                    
                    # Clip gradient if necessary
                    projected_grad = self.clip_gradient(projected_grad)

                    # Set projected gradient and optimize
                    self.set_gradient(projected_grad)
                    self.optimizer.step()
                else:
                    # No valid memory samples, just do BGD update
                    current_grad = self.clip_gradient(current_grad)
                    self.set_gradient(current_grad)
                    self.optimizer.step()
            except Exception as e:
                # Fallback to BGD-only update if memory processing fails
                print(f"Memory processing failed, using BGD only: {e}")
                current_grad = self.clip_gradient(current_grad)
                self.set_gradient(current_grad)
                self.optimizer.step()
        else:
            # No memory samples, just do BGD update
            current_grad = self.clip_gradient(current_grad)
            self.set_gradient(current_grad)
            self.optimizer.step()

        return current_loss

    def get_task_statistics(self) -> Dict:
        """Get statistics about task gradients for monitoring"""
        stats = {}
        for task_id, grad in self.task_gradients.items():
            stats[task_id] = {
                'gradient_norm': torch.norm(grad).item(),
                'sample_count': self.task_counts[task_id],
                'gradient_mean': grad.mean().item(),
                'gradient_std': grad.std().item()
            }
        return stats

    def reset_task_statistics(self):
        """Reset task gradient statistics"""
        self.task_gradients.clear()
        self.task_counts.clear()
        self.running_grad_stats.clear()

    def set_bgd_parameters(self, alpha=None, temperature=None, threshold=None):
        """Update BGD parameters during training"""
        if alpha is not None:
            self.bgd_alpha = alpha
        if temperature is not None:
            self.bgd_temperature = temperature
        if threshold is not None:
            self.gradient_similarity_threshold = threshold