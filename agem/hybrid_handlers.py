import torch
import math


class AdaptiveHybridHandler:
    """
    Hybrid approach that adaptively switches between MEGA-I and A-GEM based on
    gradient alignment and loss dynamics. Uses A-GEM when gradients are well-aligned,
    MEGA-I when they conflict but losses suggest balancing is needed.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            criterion,
            optimizer,
            device,
            batch_size=256,
            lr_scheduler=None,
            epsilon=0.1,
            alignment_threshold=0.1,  # Cosine similarity threshold for switching
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.eps_mem_batch = batch_size
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.epsilon = epsilon
        self.alignment_threshold = alignment_threshold

    def compute_gradient(self, data, labels):
        """Compute gradients for given data and labels without corrupting model state"""
        data, labels = data.to(self.device), labels.to(self.device)

        current_grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                current_grads.append(param.grad.clone())
            else:
                current_grads.append(None)

        self.model.zero_grad()
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        loss.backward()

        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))

        for param, original_grad in zip(self.model.parameters(), current_grads):
            if original_grad is not None:
                param.grad = original_grad
            else:
                param.grad = None

        return (
                   torch.cat(grads) if grads else torch.tensor([]).to(self.device)
               ), loss.item()

    def compute_cosine_similarity(self, grad1, grad2):
        """Compute cosine similarity between two gradients"""
        if grad1.numel() == 0 or grad2.numel() == 0:
            return 0.0

        norm1 = torch.norm(grad1)
        norm2 = torch.norm(grad2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return torch.dot(grad1, grad2) / (norm1 * norm2)

    def agem_projection(self, current_grad, ref_grad):
        """A-GEM style gradient projection"""
        if ref_grad.numel() == 0 or current_grad.numel() == 0:
            return current_grad

        dot_product = torch.dot(current_grad, ref_grad)
        if dot_product < 0:
            ref_grad_norm_sq = torch.dot(ref_grad, ref_grad)
            if ref_grad_norm_sq > 0:
                return current_grad - (dot_product / ref_grad_norm_sq) * ref_grad
        return current_grad

    def mega_i_balance(self, current_grad, ref_grad, current_loss, ref_loss):
        """MEGA-I style loss-based balancing"""
        if current_grad.numel() == 0 or ref_grad.numel() == 0:
            return current_grad

        if current_loss > self.epsilon:
            alpha1 = 1.0
            alpha2 = ref_loss / current_loss if current_loss > 0 else 0.0
        else:
            alpha1 = 0.0
            alpha2 = 1.0

        return alpha1 * current_grad + alpha2 * ref_grad

    def adaptive_gradient_combination(self, current_grad, ref_grad, current_loss, ref_loss):
        """Adaptively choose between A-GEM and MEGA-I based on gradient alignment"""
        cosine_sim = self.compute_cosine_similarity(current_grad, ref_grad)

        # If gradients are well-aligned (positive similarity above threshold), use A-GEM
        if cosine_sim > self.alignment_threshold:
            return self.agem_projection(current_grad, ref_grad)
        # If gradients are misaligned, use MEGA-I for better loss balancing
        else:
            return self.mega_i_balance(current_grad, ref_grad, current_loss, ref_loss)

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
        """Adaptive hybrid optimization"""
        data, labels = data.to(self.device), labels.to(self.device)

        self.model.zero_grad()
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        current_loss = loss.item()

        if self.lr_scheduler is not None:
            new_lr = self.lr_scheduler.step(current_loss)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

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

        if memory_samples is not None and len(memory_samples) > 0:
            try:
                mem_data_list = []
                mem_labels_list = []

                for sample_item in memory_samples[: self.eps_mem_batch]:
                    if isinstance(sample_item, (list, tuple)) and len(sample_item) >= 2:
                        sample_data, sample_label = sample_item[0], sample_item[1]
                    else:
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

                    ref_grad, ref_loss = self.compute_gradient(mem_data, mem_labels)

                    # Apply adaptive hybrid approach
                    hybrid_grad = self.adaptive_gradient_combination(
                        current_grad, ref_grad, current_loss, ref_loss
                    )

                    self.set_gradient(hybrid_grad)
                    self.optimizer.step()
                else:
                    self.optimizer.step()
            except Exception as e:
                print(f"Memory processing failed: {e}")
                self.optimizer.step()
        else:
            self.optimizer.step()

        return current_loss


class ProgressiveHybridHandler:
    """
    Progressive hybrid that starts with A-GEM for early stability and gradually
    incorporates more MEGA-I behavior as training progresses and losses stabilize.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            criterion,
            optimizer,
            device,
            batch_size=256,
            lr_scheduler=None,
            epsilon=0.1,
            transition_steps=1000,  # Steps to transition from A-GEM to MEGA-I
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.eps_mem_batch = batch_size
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.epsilon = epsilon
        self.transition_steps = transition_steps
        self.step_count = 0

    def compute_gradient(self, data, labels):
        """Compute gradients for given data and labels without corrupting model state"""
        data, labels = data.to(self.device), labels.to(self.device)

        current_grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                current_grads.append(param.grad.clone())
            else:
                current_grads.append(None)

        self.model.zero_grad()
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        loss.backward()

        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))

        for param, original_grad in zip(self.model.parameters(), current_grads):
            if original_grad is not None:
                param.grad = original_grad
            else:
                param.grad = None

        return (
                   torch.cat(grads) if grads else torch.tensor([]).to(self.device)
               ), loss.item()

    def agem_projection(self, current_grad, ref_grad):
        """A-GEM style gradient projection"""
        if ref_grad.numel() == 0 or current_grad.numel() == 0:
            return current_grad

        dot_product = torch.dot(current_grad, ref_grad)
        if dot_product < 0:
            ref_grad_norm_sq = torch.dot(ref_grad, ref_grad)
            if ref_grad_norm_sq > 0:
                return current_grad - (dot_product / ref_grad_norm_sq) * ref_grad
        return current_grad

    def mega_i_balance(self, current_grad, ref_grad, current_loss, ref_loss):
        """MEGA-I style loss-based balancing"""
        if current_grad.numel() == 0 or ref_grad.numel() == 0:
            return current_grad

        if current_loss > self.epsilon:
            alpha1 = 1.0
            alpha2 = ref_loss / current_loss if current_loss > 0 else 0.0
        else:
            alpha1 = 0.0
            alpha2 = 1.0

        return alpha1 * current_grad + alpha2 * ref_grad

    def progressive_gradient_combination(self, current_grad, ref_grad, current_loss, ref_loss):
        """Progressively transition from A-GEM to MEGA-I"""
        # Compute transition weight (0 = pure A-GEM, 1 = pure MEGA-I)
        transition_weight = min(self.step_count / self.transition_steps, 1.0)

        # Apply smooth transition using sigmoid for more natural blending
        smooth_weight = 1 / (1 + math.exp(-10 * (transition_weight - 0.5)))

        agem_grad = self.agem_projection(current_grad, ref_grad)
        mega_i_grad = self.mega_i_balance(current_grad, ref_grad, current_loss, ref_loss)

        # Blend the two approaches
        blended_grad = (1 - smooth_weight) * agem_grad + smooth_weight * mega_i_grad

        return blended_grad

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
        """Progressive hybrid optimization"""
        data, labels = data.to(self.device), labels.to(self.device)

        self.model.zero_grad()
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        current_loss = loss.item()

        if self.lr_scheduler is not None:
            new_lr = self.lr_scheduler.step(current_loss)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

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

        if memory_samples is not None and len(memory_samples) > 0:
            try:
                mem_data_list = []
                mem_labels_list = []

                for sample_item in memory_samples[: self.eps_mem_batch]:
                    if isinstance(sample_item, (list, tuple)) and len(sample_item) >= 2:
                        sample_data, sample_label = sample_item[0], sample_item[1]
                    else:
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

                    ref_grad, ref_loss = self.compute_gradient(mem_data, mem_labels)

                    # Apply progressive hybrid approach
                    hybrid_grad = self.progressive_gradient_combination(
                        current_grad, ref_grad, current_loss, ref_loss
                    )

                    self.set_gradient(hybrid_grad)
                    self.optimizer.step()
                else:
                    self.optimizer.step()
            except Exception as e:
                print(f"Memory processing failed: {e}")
                self.optimizer.step()
        else:
            self.optimizer.step()

        self.step_count += 1
        return current_loss


class DualCriteriaHybridHandler:
    """
    Uses both A-GEM's constraint satisfaction AND MEGA-I's loss balancing simultaneously.
    First applies A-GEM projection to ensure no negative interference, then applies
    MEGA-I style balancing within the constraint-satisfying subspace.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            criterion,
            optimizer,
            device,
            batch_size=256,
            lr_scheduler=None,
            epsilon=0.1,
            balance_strength=0.5,  # How much to emphasize loss balancing after projection
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.eps_mem_batch = batch_size
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.epsilon = epsilon
        self.balance_strength = balance_strength

    def compute_gradient(self, data, labels):
        """Compute gradients for given data and labels without corrupting model state"""
        data, labels = data.to(self.device), labels.to(self.device)

        current_grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                current_grads.append(param.grad.clone())
            else:
                current_grads.append(None)

        self.model.zero_grad()
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        loss.backward()

        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))

        for param, original_grad in zip(self.model.parameters(), current_grads):
            if original_grad is not None:
                param.grad = original_grad
            else:
                param.grad = None

        return (
                   torch.cat(grads) if grads else torch.tensor([]).to(self.device)
               ), loss.item()

    def dual_criteria_combination(self, current_grad, ref_grad, current_loss, ref_loss):
        """Apply both A-GEM projection and MEGA-I balancing"""
        if current_grad.numel() == 0 or ref_grad.numel() == 0:
            return current_grad

        # Step 1: Apply A-GEM projection to ensure constraint satisfaction
        dot_product = torch.dot(current_grad, ref_grad)
        projected_grad = current_grad

        if dot_product < 0:
            ref_grad_norm_sq = torch.dot(ref_grad, ref_grad)
            if ref_grad_norm_sq > 0:
                projected_grad = current_grad - (dot_product / ref_grad_norm_sq) * ref_grad

        # Step 2: Apply MEGA-I style balancing within the constraint-satisfying subspace
        if current_loss > self.epsilon:
            alpha1 = 1.0
            alpha2 = (ref_loss / current_loss) * self.balance_strength if current_loss > 0 else 0.0
        else:
            alpha1 = 1.0 - self.balance_strength
            alpha2 = self.balance_strength

        # Combine projected gradient with reference gradient for balancing
        balanced_grad = alpha1 * projected_grad + alpha2 * ref_grad

        # Ensure the final gradient still satisfies A-GEM constraint
        final_dot_product = torch.dot(balanced_grad, ref_grad)
        if final_dot_product < 0:
            # If balancing violated constraint, fall back to projected gradient
            return projected_grad

        return balanced_grad

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
        """Dual criteria hybrid optimization"""
        data, labels = data.to(self.device), labels.to(self.device)

        self.model.zero_grad()
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        current_loss = loss.item()

        if self.lr_scheduler is not None:
            new_lr = self.lr_scheduler.step(current_loss)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

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

        if memory_samples is not None and len(memory_samples) > 0:
            try:
                mem_data_list = []
                mem_labels_list = []

                for sample_item in memory_samples[: self.eps_mem_batch]:
                    if isinstance(sample_item, (list, tuple)) and len(sample_item) >= 2:
                        sample_data, sample_label = sample_item[0], sample_item[1]
                    else:
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

                    ref_grad, ref_loss = self.compute_gradient(mem_data, mem_labels)

                    # Apply dual criteria approach
                    hybrid_grad = self.dual_criteria_combination(
                        current_grad, ref_grad, current_loss, ref_loss
                    )

                    self.set_gradient(hybrid_grad)
                    self.optimizer.step()
                else:
                    self.optimizer.step()
            except Exception as e:
                print(f"Memory processing failed: {e}")
                self.optimizer.step()
        else:
            self.optimizer.step()

        return current_loss
