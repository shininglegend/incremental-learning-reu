import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random

class AGEMHandler:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion,
        optimizer,
        device,
        batch_size=256,
        lr_scheduler=None,
        # New clustering parameters
        n_clusters=8,
        cluster_update_freq=100,
        min_cluster_samples=5,
        max_cluster_samples=64,
        importance_decay=0.95,
        diversity_weight=0.3,
        gradient_weighting_strategy='importance',  # 'importance', 'size', 'uniform', 'mixed'
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.base_mem_batch = batch_size
        self.device = device
        self.lr_scheduler = lr_scheduler
        
        # Clustering parameters
        self.n_clusters = n_clusters
        self.cluster_update_freq = cluster_update_freq
        self.min_cluster_samples = min_cluster_samples
        self.max_cluster_samples = max_cluster_samples
        self.importance_decay = importance_decay
        self.diversity_weight = diversity_weight
        self.gradient_weighting_strategy = gradient_weighting_strategy
        
        # Clustering state
        self.cluster_model = None
        self.cluster_assignments = {}
        self.cluster_centroids = {}
        self.cluster_importance = {}
        self.sample_importance = {}
        self.update_counter = 0
        
        # Memory organization
        self.clustered_memory = defaultdict(list)
        self.memory_features = {}

    def extract_features(self, data):
        """Extract features from data for clustering"""
        self.model.eval()
        with torch.no_grad():
            # Use intermediate layer features or final layer before classification
            if hasattr(self.model, 'feature_extractor'):
                features = self.model.feature_extractor(data)
            else:
                # Fallback: use flattened input or model output
                features = data.view(data.size(0), -1)
        self.model.train()
        return features.detach().cpu().numpy()

    def update_clusters(self, memory_samples):
        """Update cluster assignments and importance weights"""
        if len(memory_samples) < self.n_clusters:
            return
        
        # Extract features from all memory samples
        all_features = []
        sample_indices = []
        
        for idx, sample_item in enumerate(memory_samples):
            try:
                if isinstance(sample_item, (list, tuple)) and len(sample_item) >= 2:
                    sample_data = sample_item[0]
                else:
                    sample_data = sample_item
                    
                # Ensure tensor is properly formatted
                if isinstance(sample_data, torch.Tensor):
                    if sample_data.dim() == 1:
                        # If 1D, assume it needs to be reshaped for your model
                        sample_data = sample_data.unsqueeze(0)
                    elif sample_data.dim() == 2 and sample_data.size(0) != 1:
                        # If 2D but not batch format, add batch dimension
                        sample_data = sample_data.unsqueeze(0)
                    elif sample_data.dim() >= 3 and sample_data.size(0) != 1:
                        # If already has batch dimension > 1, take first sample
                        sample_data = sample_data[:1]
                    
                    features = self.extract_features(sample_data.to(self.device))
                    all_features.append(features.flatten())
                    sample_indices.append(idx)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        if len(all_features) < self.n_clusters:
            return
            
        # Perform clustering
        features_array = np.array(all_features)
        
        # Handle case where features might be empty or invalid
        if features_array.shape[0] == 0 or features_array.shape[1] == 0:
            return
            
        try:
            self.cluster_model = KMeans(n_clusters=min(self.n_clusters, len(all_features)), 
                                      random_state=42, n_init=10)
            cluster_labels = self.cluster_model.fit_predict(features_array)
        except Exception as e:
            print(f"Clustering failed: {e}")
            return
        
        # Update cluster assignments
        self.cluster_assignments = {}
        self.clustered_memory = defaultdict(list)
        
        for sample_idx, cluster_id in zip(sample_indices, cluster_labels):
            self.cluster_assignments[sample_idx] = cluster_id
            self.clustered_memory[cluster_id].append(sample_idx)
            
        # Update cluster centroids
        self.cluster_centroids = {}
        for cluster_id in range(self.cluster_model.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = features_array[cluster_mask]
            if len(cluster_features) > 0:
                self.cluster_centroids[cluster_id] = np.mean(cluster_features, axis=0)
        
        # Update importance weights
        self.update_importance_weights(memory_samples, features_array, cluster_labels)

    def update_importance_weights(self, memory_samples, features_array, cluster_labels):
        """Update importance weights based on cluster properties"""
        # Initialize importance weights
        self.cluster_importance = {}
        self.sample_importance = {}
        
        n_clusters = len(np.unique(cluster_labels))
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = features_array[cluster_mask]
            cluster_size = len(cluster_features)
            
            if cluster_size == 0:
                continue
                
            # Cluster importance based on size and diversity
            if cluster_size > 1:
                cluster_diversity = np.std(cluster_features, axis=0).mean()
            else:
                cluster_diversity = 0.0
                
            size_weight = 1.0 / (1.0 + cluster_size)  # Smaller clusters get higher weight
            diversity_weight = cluster_diversity
            
            cluster_importance = size_weight * (1 - self.diversity_weight) + diversity_weight * self.diversity_weight
            self.cluster_importance[cluster_id] = cluster_importance
            
            # Sample importance within cluster (distance to centroid)
            if cluster_id in self.cluster_centroids:
                centroid = self.cluster_centroids[cluster_id]
                cluster_indices = np.where(cluster_mask)[0]
                
                for local_idx in range(len(cluster_indices)):
                    try:
                        # Get the actual sample index from our tracking
                        global_sample_idx = None
                        samples_in_cluster = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                        if local_idx < len(samples_in_cluster):
                            global_sample_idx = samples_in_cluster[local_idx]
                        
                        if global_sample_idx is not None and local_idx < len(cluster_features):
                            # Distance to centroid (closer samples are more representative)
                            distance = np.linalg.norm(cluster_features[local_idx] - centroid)
                            representativeness = 1.0 / (1.0 + distance)
                            
                            # Combine cluster and sample importance
                            total_importance = cluster_importance * representativeness
                            self.sample_importance[global_sample_idx] = total_importance
                    except Exception as e:
                        print(f"Error computing sample importance: {e}")
                        continue

    def compute_dynamic_batch_size(self, memory_samples):
        """Compute adaptive batch size based on cluster distribution"""
        if not self.cluster_assignments:
            return self.base_mem_batch
            
        # Count samples per cluster
        cluster_counts = defaultdict(int)
        for sample_idx in self.cluster_assignments:
            if sample_idx < len(memory_samples):
                cluster_id = self.cluster_assignments[sample_idx]
                cluster_counts[cluster_id] += 1
        
        # Ensure minimum representation from each cluster
        active_clusters = len([c for c in cluster_counts.values() if c > 0])
        if active_clusters == 0:
            return self.base_mem_batch
            
        # Dynamic sizing: ensure each cluster gets at least min_cluster_samples
        min_required = active_clusters * self.min_cluster_samples
        max_allowed = active_clusters * self.max_cluster_samples
        
        # Adapt between min and max based on total available samples
        total_samples = len(memory_samples)
        dynamic_size = min(max(min_required, self.base_mem_batch), 
                          min(max_allowed, total_samples))
        
        return dynamic_size

    def select_importance_weighted_samples(self, memory_samples, target_size):
        """Select samples using importance weighting and cluster-aware strategy"""
        if not self.cluster_assignments or not self.sample_importance:
            # Fallback to random selection
            return memory_samples[:target_size], None
        
        # Group samples by cluster
        cluster_samples = defaultdict(list)
        for sample_idx in range(len(memory_samples)):
            if sample_idx in self.cluster_assignments:
                cluster_id = self.cluster_assignments[sample_idx]
                importance = self.sample_importance.get(sample_idx, 0.1)
                cluster_samples[cluster_id].append((sample_idx, importance))
        
        # Allocate samples per cluster based on cluster importance
        selected_samples = []
        selected_weights = []
        selected_indices = set()  # Track selected indices to avoid duplicates
        remaining_budget = target_size
        
        # Sort clusters by importance
        sorted_clusters = sorted(self.cluster_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        total_cluster_importance = sum(imp for _, imp in sorted_clusters)
        if total_cluster_importance == 0:
            total_cluster_importance = 1  # Avoid division by zero
        
        for cluster_id, cluster_importance in sorted_clusters:
            if remaining_budget <= 0:
                break
                
            cluster_sample_list = cluster_samples[cluster_id]
            if not cluster_sample_list:
                continue
                
            # Allocate samples to this cluster
            cluster_allocation = max(
                self.min_cluster_samples,
                min(self.max_cluster_samples, 
                    int(remaining_budget * cluster_importance / total_cluster_importance))
            )
            cluster_allocation = min(cluster_allocation, len(cluster_sample_list), remaining_budget)
            
            # Select top samples from this cluster by importance
            cluster_sample_list.sort(key=lambda x: x[1], reverse=True)
            for i in range(cluster_allocation):
                sample_idx, importance = cluster_sample_list[i]
                if sample_idx not in selected_indices:
                    selected_samples.append(memory_samples[sample_idx])
                    selected_weights.append(importance)
                    selected_indices.add(sample_idx)
                    remaining_budget -= 1
        
        # Fill remaining slots with highest importance samples overall
        if remaining_budget > 0:
            all_remaining = []
            for cluster_samples_list in cluster_samples.values():
                all_remaining.extend(cluster_samples_list)
            
            all_remaining.sort(key=lambda x: x[1], reverse=True)
            for i in range(min(remaining_budget, len(all_remaining))):
                sample_idx, importance = all_remaining[i]
                if sample_idx not in selected_indices:
                    selected_samples.append(memory_samples[sample_idx])
                    selected_weights.append(importance)
                    selected_indices.add(sample_idx)
                    remaining_budget -= 1
        
        # Normalize weights to sum to 1
        if selected_weights:
            weight_tensor = torch.tensor(selected_weights, dtype=torch.float32)
            weight_tensor = weight_tensor / weight_tensor.sum()
            return selected_samples, weight_tensor
        
        return selected_samples, None

    def compute_gradient(self, data, labels, sample_weights=None):
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
        
        # Compute loss with optional sample weights
        if sample_weights is not None:
            sample_weights = sample_weights.to(self.device)
            # Compute individual losses for each sample
            individual_losses = self.criterion(outputs, labels)
            if individual_losses.dim() == 0:
                # If criterion returns scalar, compute individual losses manually
                individual_losses = torch.nn.functional.cross_entropy(
                    outputs, labels, reduction='none'
                )
            # Weight the losses by sample importance
            weighted_loss = torch.mean(individual_losses * sample_weights)
            loss = weighted_loss
        else:
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

    def compute_cluster_weighted_gradient(self, data, labels, cluster_ids, weighting_strategy='importance'):
        """Compute gradients with advanced cluster-based weighting strategies"""
        # Move data to device
        data, labels = data.to(self.device), labels.to(self.device)
        
        # Save current gradients
        current_grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                current_grads.append(param.grad.clone())
            else:
                current_grads.append(None)

        # Compute gradients for each cluster separately
        cluster_gradients = []
        cluster_weights = []
        
        unique_clusters = torch.unique(cluster_ids)
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_ids == cluster_id
            cluster_data = data[cluster_mask]
            cluster_labels = labels[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
                
            # Compute gradient for this cluster
            self.model.zero_grad()
            outputs = self.model(cluster_data)
            loss = self.criterion(outputs, cluster_labels)
            loss.backward()
            
            # Extract cluster gradient
            cluster_grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    cluster_grad.append(param.grad.view(-1))
            
            if cluster_grad:
                cluster_gradients.append(torch.cat(cluster_grad))
                
                # Compute cluster weight based on strategy
                cluster_id_item = cluster_id.item()
                if weighting_strategy == 'importance':
                    weight = self.cluster_importance.get(cluster_id_item, 1.0)
                elif weighting_strategy == 'size':
                    weight = 1.0 / len(cluster_data)  # Inverse size weighting
                elif weighting_strategy == 'uniform':
                    weight = 1.0
                else:
                    weight = 1.0
                    
                cluster_weights.append(weight)
        
        # Combine cluster gradients with weights
        if cluster_gradients:
            # Normalize weights
            total_weight = sum(cluster_weights)
            if total_weight > 0:
                cluster_weights = [w / total_weight for w in cluster_weights]
            else:
                cluster_weights = [1.0 / len(cluster_weights)] * len(cluster_weights)
            
            # Weighted sum of cluster gradients
            combined_grad = torch.zeros_like(cluster_gradients[0])
            for grad, weight in zip(cluster_gradients, cluster_weights):
                combined_grad += weight * grad
        else:
            combined_grad = torch.tensor([]).to(self.device)
        
        # Restore original gradients
        for param, original_grad in zip(self.model.parameters(), current_grads):
            if original_grad is not None:
                param.grad = original_grad
            else:
                param.grad = None
                
        return combined_grad

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
        """Enhanced A-GEM optimization with cluster-aware memory selection"""
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
        current_grad = torch.cat(current_grad) if current_grad else torch.tensor([])

        # Enhanced memory processing with clustering
        if memory_samples is not None and len(memory_samples) > 0:
            try:
                # Update clusters periodically
                self.update_counter += 1
                if self.update_counter % self.cluster_update_freq == 0:
                    self.update_clusters(memory_samples)
                
                # Compute dynamic batch size
                dynamic_batch_size = self.compute_dynamic_batch_size(memory_samples)
                
                # Select importance-weighted samples - FIXED: unpack the tuple
                selected_samples, sample_weights = self.select_importance_weighted_samples(
                    memory_samples, dynamic_batch_size
                )
                
                # Prepare memory data
                mem_data_list = []
                mem_labels_list = []

                for sample_item in selected_samples:
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

                    # Compute reference gradient on selected memory with optional weights
                    ref_grad = self.compute_gradient(mem_data, mem_labels, sample_weights)

                    # Project current gradient
                    projected_grad = self.project_gradient(current_grad, ref_grad)

                    # Set projected gradient and optimize
                    self.set_gradient(projected_grad)
                    self.optimizer.step()
                else:
                    self.optimizer.step()
                    
            except Exception as e:
                print(f"Cluster-aware memory processing failed: {e}")
                self.optimizer.step()
        else:
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
