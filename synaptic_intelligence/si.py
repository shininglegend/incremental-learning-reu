# Continual Learning through Synaptic Intelligence 
# https://proceedings.mlr.press/v70/zenke17a 

import torch


class SynapticIntelligence:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion,
        optimizer,
        device,
        lr_scheduler=None,
        si_c=0.1,  # SI regularization strength
        xi=1.0,  # SI damping parameter
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.lr_scheduler = lr_scheduler

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

    def optimize(self, data, labels, memory_samples=None):
        """SI optimization with regularization"""
        # Move data to device
        data, labels = data.to(self.device), labels.to(self.device)

        # Compute loss
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

        # Backpropagate and optimize
        total_loss.backward()

        # Update SI gradient accumulation
        self.update_si_gradients()

        self.optimizer.step()

        # Update SI parameter deltas after optimization step
        self.update_si_deltas()

        return current_loss

    def end_task(self):
        """Call this method when finishing a task to update SI importance weights"""
        print("Computing Synaptic Intelligence importance weights...")
        self.compute_si_importance()
        print("SI importance weights updated.")
