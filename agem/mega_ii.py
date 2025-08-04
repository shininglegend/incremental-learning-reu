import torch
import math


class MEGA2Handler:
    def __init__(
            self,
            model: torch.nn.Module,
            criterion,
            optimizer,
            device,
            batch_size=256,
            lr_scheduler=None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.eps_mem_batch = batch_size
        self.device = device
        self.lr_scheduler = lr_scheduler

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

        return (
                   torch.cat(grads) if grads else torch.tensor([]).to(self.device)
               ), loss.item()

    def compute_loss(self, data, labels):
        """Compute loss for given data and labels without affecting gradients"""
        data, labels = data.to(self.device), labels.to(self.device)

        with torch.no_grad():
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)

        return loss.item()

    def mega_ii_gradient_rotation(self, current_grad, ref_grad, current_loss, ref_loss):
        """
        Rotate current gradient using MEGA-II approach based on loss-balanced cosine similarity.

        According to the paper:
        - Find optimal angle θ to maximize: l_t * cos(β) + l_ref * cos(θ̃ - β)
        - Where θ̃ is angle between current and reference gradients
        - β is angle between mixed gradient and current gradient
        """
        if current_grad.numel() == 0 or ref_grad.numel() == 0:
            return current_grad

        # Normalize gradients for angle computation
        current_grad_norm = torch.norm(current_grad)
        ref_grad_norm = torch.norm(ref_grad)

        if current_grad_norm == 0 or ref_grad_norm == 0:
            return current_grad

        current_grad_unit = current_grad / current_grad_norm
        ref_grad_unit = ref_grad / ref_grad_norm

        # Compute angle between current and reference gradients
        cos_theta_tilde = torch.clamp(torch.dot(current_grad_unit, ref_grad_unit), -1.0, 1.0)
        theta_tilde = torch.acos(cos_theta_tilde)

        # Handle special cases
        if ref_loss == 0:
            # No catastrophic forgetting, use current gradient
            return current_grad

        if current_loss == 0:
            # Perfect on current task, rotate to reference direction
            # Scale reference gradient to match current gradient magnitude
            return (current_grad_norm / ref_grad_norm) * ref_grad

        # Compute loss ratio k = l_t / l_ref
        k = current_loss / ref_loss if ref_loss > 0 else float('inf')

        # Handle case where gradients are parallel or anti-parallel
        if theta_tilde < 1e-6:  # Nearly parallel
            return current_grad
        elif theta_tilde > math.pi - 1e-6:  # Nearly anti-parallel
            # Project current gradient as in A-GEM
            dot_product = torch.dot(current_grad, ref_grad)
            if dot_product < 0:
                ref_grad_norm_sq = torch.dot(ref_grad, ref_grad)
                if ref_grad_norm_sq > 0:
                    return current_grad - (dot_product / ref_grad_norm_sq) * ref_grad
            return current_grad

        # Compute optimal rotation angle β
        # From paper: α = arctan((k + cos(θ̃)) / sin(θ̃)) and β = π/2 - α
        sin_theta_tilde = torch.sin(theta_tilde)

        if sin_theta_tilde < 1e-6:
            return current_grad

        alpha = torch.atan((k + cos_theta_tilde) / sin_theta_tilde)
        beta = math.pi / 2 - alpha

        # Clamp β to valid range [0, π]
        beta = torch.clamp(beta, 0, math.pi)

        # Create rotation matrix in 2D subspace spanned by current_grad and ref_grad
        # Mixed gradient: g_mix = cos(β) * current_grad_unit + sin(β) * perpendicular_component

        # Find component of ref_grad perpendicular to current_grad
        ref_parallel = torch.dot(ref_grad_unit, current_grad_unit) * current_grad_unit
        ref_perpendicular = ref_grad_unit - ref_parallel
        ref_perpendicular_norm = torch.norm(ref_perpendicular)

        if ref_perpendicular_norm < 1e-6:
            # Gradients are parallel, no rotation needed
            return current_grad

        ref_perpendicular_unit = ref_perpendicular / ref_perpendicular_norm

        # Compute mixed gradient direction
        cos_beta = torch.cos(beta)
        sin_beta = torch.sin(beta)

        mixed_grad_unit = cos_beta * current_grad_unit + sin_beta * ref_perpendicular_unit

        # Scale to maintain current gradient magnitude
        mixed_grad = current_grad_norm * mixed_grad_unit

        return mixed_grad

    def set_gradient(self, grad_vector):
        """Set model gradients from flattened gradient vector"""
        if grad_vector.numel() == 0:
            return

        pointer = 0
        for param in self.model.parameters():
            if param.grad is not None:
                num_param = param.numel()
                param.grad.data = grad_vector[pointer: pointer + num_param].view(
                    param.shape
                )
                pointer += num_param

    def optimize(self, data, labels, memory_samples=None):
        """MEGA-II optimization with loss-based gradient rotation"""
        data, labels = data.to(self.device), labels.to(self.device)

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
        current_grad = (
            torch.cat(current_grad)
            if current_grad
            else torch.tensor([]).to(self.device)
        )

        # If we have memory samples, apply MEGA-II gradient rotation
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

                    # Apply MEGA-II gradient rotation
                    rotated_grad = self.mega_ii_gradient_rotation(
                        current_grad, ref_grad, current_loss, ref_loss
                    )

                    # Set rotated gradient and optimize
                    self.set_gradient(rotated_grad)
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