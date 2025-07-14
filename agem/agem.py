import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from collections import deque
import random

class EpisodicMemory:
    """Stores samples from previous tasks for reference loss computation"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        
    def add(self, x: torch.Tensor, y: torch.Tensor, task_id: int):
        """Add a sample to episodic memory"""
        self.memory.append((x.clone(), y.clone(), task_id))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Sample a batch from episodic memory"""
        if len(self.memory) == 0:
            return None, None, []
        
        batch_size = min(batch_size, len(self.memory))
        samples = random.sample(self.memory, batch_size)
        
        x_batch = torch.stack([s[0] for s in samples])
        y_batch = torch.stack([s[1] for s in samples])
        task_ids = [s[2] for s in samples]
        
        return x_batch, y_batch, task_ids

class AGEMHandler:
    """
    A-GEM Handler with integrated MEGA-I (Mixed Episodic Gradient Averaging with Importance)
    Combines gradient projection with adaptive loss-based weighting
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        criterion,
        optimizer,
        device,
        batch_size=256,
        lr_scheduler=None,
        epsilon=0.1,  # MEGA-I threshold parameter
        memory_capacity=1000,
        use_projection=True,  # Whether to use A-GEM projection
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.eps_mem_batch = batch_size  # Memory batch size for gradient computation
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.epsilon = epsilon
        self.use_projection = use_projection
        
        # Initialize episodic memory
        self.memory = EpisodicMemory(memory_capacity)
        
        # Tracking variables
        self.current_task_id = 0
        self.loss_history = []
        self.alpha_history = []

    def compute_current_task_loss(self, data, labels):
        """Compute loss on current task data"""
        data, labels = data.to(self.device), labels.to(self.device)
        
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        return loss

    def compute_reference_loss(self, batch_size=None):
        """Compute reference loss from episodic memory using MEGA-I approach"""
        if batch_size is None:
            batch_size = self.eps_mem_batch
            
        x_ref, y_ref, _ = self.memory.sample(batch_size)
        
        if x_ref is None:
            return torch.tensor(0.0).to(self.device)
        
        x_ref, y_ref = x_ref.to(self.device), y_ref.to(self.device)
        outputs = self.model(x_ref)
        return self.criterion(outputs, y_ref)

    def compute_alpha_weights_mega_i(self, current_loss, reference_loss):
        """
        Compute adaptive weights α₁ and α₂ based on MEGA-I equations (6)
        
        Case 1: ℓt(w; ξ) > ε
            α₁(w) = 1, α₂(w) = ℓref(w; ξ)/ℓt(w; ξ)
        Case 2: ℓt(w; ξ) ≤ ε  
            α₁(w) = 0, α₂(w) = 1
        """
        current_loss_val = current_loss.item() if isinstance(current_loss, torch.Tensor) else current_loss
        reference_loss_val = reference_loss.item() if isinstance(reference_loss, torch.Tensor) else reference_loss
        
        if current_loss_val > self.epsilon:
            # Case 1: High current loss - focus on current task
            alpha_1 = 1.0
            alpha_2 = reference_loss_val / current_loss_val if current_loss_val > 0 else 0.0
        else:
            # Case 2: Low current loss - focus on memory consolidation
            alpha_1 = 0.0
            alpha_2 = 1.0
            
        return alpha_1, alpha_2

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

    def compute_mixed_gradient(self, current_grad, ref_grad, alpha_1, alpha_2):
        """
        Compute mixed gradient using MEGA-I weighting scheme
        """
        if ref_grad.numel() == 0 or current_grad.numel() == 0:
            return current_grad
        
        # Compute weighted combination of gradients
        mixed_grad = alpha_1 * current_grad + alpha_2 * ref_grad
        return mixed_grad

    def project_gradient(self, current_grad, ref_grad):
        """Project current gradient to not increase loss on reference gradient (A-GEM)"""
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

    #####
    def train_step(self, data, labels, store_in_memory=True):
        """
        Single training step with MEGA-I adaptive loss balancing and optional A-GEM projection
        """
        # Move data to device
        data, labels = data.to(self.device), labels.to(self.device)
        
        # Compute current task loss
        current_loss = self.compute_current_task_loss(data, labels)
        
        # Update learning rate if scheduler is provided
        if self.lr_scheduler is not None:
            new_lr = self.lr_scheduler.step(current_loss.item())
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

        # Compute reference loss from episodic memory
        reference_loss = self.compute_reference_loss()
        
        # Compute adaptive weights using MEGA-I
        alpha_1, alpha_2 = self.compute_alpha_weights_mega_i(current_loss, reference_loss)
        
        # Compute current task gradient
        self.model.zero_grad()
        current_loss.backward()
        current_grad = []
        for param in self.model.parameters():
            if param.grad is not None:
                current_grad.append(param.grad.view(-1))
        current_grad = torch.cat(current_grad) if current_grad else torch.tensor([]).to(self.device)

        # Process with memory if available
        if len(self.memory.memory) > 0:
            # Get memory samples for gradient computation
            x_ref, y_ref, _ = self.memory.sample(self.eps_mem_batch)
            
            if x_ref is not None:
                # Compute reference gradient
                ref_grad = self.compute_gradient(x_ref, y_ref)
                
                # Compute mixed gradient using MEGA-I weights
                mixed_grad = self.compute_mixed_gradient(current_grad, ref_grad, alpha_1, alpha_2)
                
                # Apply A-GEM projection if enabled
                if self.use_projection:
                    final_grad = self.project_gradient(mixed_grad, ref_grad)
                else:
                    final_grad = mixed_grad
                
                # Set final gradient and optimize
                self.set_gradient(final_grad)
                self.optimizer.step()
            else:
                # No valid memory samples, just do regular update
                self.optimizer.step()
        else:
            # No memory samples, just do regular update
            self.optimizer.step()

        # Store samples in episodic memory
        if store_in_memory:
            for i in range(data.size(0)):
                self.memory.add(data[i], labels[i], self.current_task_id)
        
        # Compute total loss for tracking
        total_loss = alpha_1 * current_loss + alpha_2 * reference_loss
        
        # Track metrics
        metrics = {
            'current_loss': current_loss.item(),
            'reference_loss': reference_loss.item(),
            'total_loss': total_loss.item(),
            'alpha_1': alpha_1,
            'alpha_2': alpha_2,
            'below_threshold': current_loss.item() <= self.epsilon
        }
        
        self.loss_history.append(metrics)
        self.alpha_history.append((alpha_1, alpha_2))
        
        return metrics

    def optimize(self, data, labels, memory_samples=None, use_mixed_gradient=True):
        """
        A-GEM optimization with Mixed Stochastic Gradient (MEGA-I) support
        
        Args:
            data: Current task data
            labels: Current task labels
            memory_samples: Episodic memory samples
            use_mixed_gradient: Whether to use mixed gradient approach (True) or original A-GEM projection (False)
        """
        # Compute current task loss
        current_loss = self.compute_current_task_loss(data, labels)
        
        # Update learning rate if scheduler is provided
        if self.lr_scheduler is not None:
            new_lr = self.lr_scheduler.step(current_loss.item())
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

        # Compute gradient on current task
        self.model.zero_grad()
        current_loss.backward()
        current_grad = []
        for param in self.model.parameters():
            if param.grad is not None:
                current_grad.append(param.grad.view(-1))
        current_grad = torch.cat(current_grad) if current_grad else torch.tensor([]).to(self.device)

        # If we have memory samples, apply mixed gradient or projection
        if memory_samples is not None and len(memory_samples) > 0:
            # Compute reference loss
            reference_loss = self.compute_reference_loss(memory_samples)
            
            # Compute reference gradient
            ref_grad = self.compute_gradient(
                torch.stack([item[0] for item in memory_samples[:self.eps_mem_batch]]).to(self.device),
                torch.tensor([item[1] for item in memory_samples[:self.eps_mem_batch]]).to(self.device)
            )
            
            if use_mixed_gradient:
                # Use Mixed Stochastic Gradient approach (MEGA-I)
                alpha_current, alpha_memory = self.compute_alpha_weights_mega_i(current_loss, reference_loss)
                
                # Compute mixed gradient
                mixed_grad = self.compute_mixed_gradient(current_grad, ref_grad, alpha_current, alpha_memory)
                
                # Set mixed gradient and optimize
                self.set_gradient(mixed_grad)
                self.optimizer.step()
            else:
                # Use original A-GEM projection
                projected_grad = self.project_gradient(current_grad, ref_grad)
                self.set_gradient(projected_grad)
                self.optimizer.step()
        else:
            # No memory samples, just do regular update
            self.optimizer.step()

        return current_loss.item()
    
    def evaluate(self, test_loader) -> float:
        """Evaluate model accuracy"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                predictions = self.model(data)
                _, predicted = torch.max(predictions.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        self.model.train()
        return correct / total
    
    def train_task(self, 
                   train_loader, 
                   epochs: int = 10,
                   task_id: int = None) -> List[Dict[str, float]]:
        """Train on a specific task"""
        if task_id is not None:
            self.current_task_id = task_id
            
        epoch_metrics = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_alpha_1 = 0
            epoch_alpha_2 = 0
            
            for batch_idx, (data, labels) in enumerate(train_loader):
                metrics = self.train_step(data, labels)
                
                epoch_loss += metrics['total_loss']
                epoch_alpha_1 += metrics['alpha_1']
                epoch_alpha_2 += metrics['alpha_2']
            
            # Average metrics for the epoch
            num_batches = len(train_loader)
            epoch_metrics.append({
                'epoch': epoch,
                'avg_loss': epoch_loss / num_batches,
                'avg_alpha_1': epoch_alpha_1 / num_batches,
                'avg_alpha_2': epoch_alpha_2 / num_batches
            })
            
            print(f"Task {self.current_task_id}, Epoch {epoch}: "
                  f"Loss={epoch_loss/num_batches:.4f}, "
                  f"α₁={epoch_alpha_1/num_batches:.3f}, "
                  f"α₂={epoch_alpha_2/num_batches:.3f}")
        
        return epoch_metrics

    def evaluate(self, test_loader) -> float:
        """Evaluate model accuracy"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                predictions = self.model(data)
                _, predicted = torch.max(predictions.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        self.model.train()
        return correct / total
    
    def get_alpha_weights(self):
        """Return current alpha weights for monitoring"""
        if self.alpha_history:
            return self.alpha_history[-1]
        return 0.5, 0.5

    def get_memory_size(self):
        """Return current episodic memory size"""
        return len(self.memory.memory)


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
