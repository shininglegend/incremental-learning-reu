# Code written by Gemini 2.5 Flash. Edited by Titus
import agem
import clustering
import mnist
# import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. Configuration and Initialization ---
NUM_CLASSES = 10 # For MNIST
INPUT_DIM = 784  # For MNIST (28*28)
HIDDEN_DIM = 200 # As per paper
MEMORY_SIZE_Q = 10 # Number of clusters (Q)
MEMORY_SIZE_P = 3  # Max samples per cluster (P)
BATCH_SIZE = 10    # As per paper
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
NUM_TASKS = 5 # Example: for permutation or rotation based tasks

# Define a simple MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(-1, INPUT_DIM) # Flatten MNIST images
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Initialize model, optimizer, and loss function
model = SimpleMLP(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Initialize TA-A-GEM components
# Clustering_memory will manage the episodic memory
clustering_memory = clustering.ClusteringMemory(
    Q=MEMORY_SIZE_Q, P=MEMORY_SIZE_P, input_type='samples' # 'samples' for TA-A-GEM
)

# A-GEM wrapper/class for gradient projection logic
# It needs to access the memory managed by clustering_memory
agem_handler = agem.AGEMHandler(model, criterion, optimizer)

# Load and prepare MNIST data for domain-incremental learning
# This function would encapsulate the permutation, rotation, or class split logic
# It returns a list of data loaders, one for each task/domain
task_dataloaders = mnist.prepare_domain_incremental_mnist(
    task_type='permutation', num_tasks=NUM_TASKS, batch_size=BATCH_SIZE
)

# --- 2. Training Loop ---
print("Starting TA-A-GEM training...")
for task_id, task_dataloader in enumerate(task_dataloaders):
    print(f"\n--- Training on Task {task_id} ---")

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, (data, labels) in enumerate(task_dataloader):
            # Step 1: Use A-GEM logic for current batch and current memory
            # agem_handler.optimize handles model update and gradient projection
            # It queries clustering_memory for the current reference samples
            agem_handler.optimize(data, labels, clustering_memory.get_memory_samples())

            # Step 2: Update the clustered memory with current batch samples
            # This is where the core clustering for TA-A-GEM happens
            for i in range(len(data)):
                sample_data = data[i]
                sample_label = labels[i]
                clustering_memory.add_sample(sample_data, sample_label) # Add sample to clusters

            if batch_idx % 100 == 0:
                print(f"Task {task_id}, Epoch {epoch}, Batch {batch_idx}")

    # Optional: Evaluate performance after each task (as in the paper's "disjoint tasks" experiments)
    model.eval()
    avg_accuracy = agem.evaluate_all_tasks(model, criterion, task_dataloaders)
    print(f"After Task {task_id}, Average Accuracy: {avg_accuracy:.4f}")

print("\nTA-A-GEM training complete.")