# This is the file pulling it all together. Edit sparingly, if at all!
import time, os
from utils import accuracy_test
from init import initialize_system

# --- 1. Configuration and Initialization ---

# Initialize system with configuration
# Parse command line arguments
(
    config,
    params,
    t,
    device,
    model,
    optimizer,
    criterion,
    lr_scheduler,
    clustering_memory,
    agem_handler,
    train_dataloaders,
    test_dataloaders,
    visualizer,
) = initialize_system()

# Extract commonly used variables from config
QUICK_TEST_MODE = config["lite"]
VERBOSE = config["verbose"]
NUM_EPOCHS = config["num_epochs"]
USE_LEARNING_RATE_SCHEDULER = config["use_learning_rate_scheduler"]
LEARNING_RATE = config["learning_rate"]
t.start("training")

# --- 2. Training Loop ---
print("Starting training...")
print(
    f"""
Quick Test mode: {QUICK_TEST_MODE} | Task Type: {config['task_type']}
Total tasks: {len(train_dataloaders)}"""
)

for task_id, train_dataloader in enumerate(train_dataloaders):
    print(f"\n--- Training on Task {task_id + 1} ---")

    task_start_time = time.time()
    task_epoch_losses = []

    for epoch in range(NUM_EPOCHS):

        # Initializes the loss normalization at the start of each epoch
        agem_handler.start_epoch()
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (data, labels) in enumerate(train_dataloader):
            # Move data to device
            data, labels = data.to(device), labels.to(device)

            # Step 1: Use A-GEM logic for current batch and current memory
            # agem_handler.optimize handles model update and gradient projection
            # It queries clustering_memory for the current reference samples
            t.start("get samples")
            _samples = clustering_memory.get_memory_samples()
            t.end("get samples")
            t.start("optimize")
            batch_loss = agem_handler.optimize(data, labels, _samples)
            t.end("optimize")

            # Track batch loss
            if batch_loss is not None:
                epoch_loss += batch_loss
                visualizer.add_batch_loss(task_id, epoch, batch_idx, batch_loss)

            # Step 2: Update the clustered memory with current batch samples
            # This is where the core clustering for TA-A-GEM happens
            t.start("add samples")
            # Add only first sample from batch
            # This duplicates the activity of the paper and helps not overwrite previous tasks too fast
            sample_data = data[0].cpu()  # Move to CPU for memory storage
            sample_label = labels[0].cpu()  # Move to CPU for memory storage
            clustering_memory.add_sample(
                sample_data, sample_label, task_id
            )  # Add sample to clusters
            t.end("add samples")

            num_batches += 1

            # Update progress bar every 50 batches or on last batch
            if (
                not QUICK_TEST_MODE
                and VERBOSE
                and (batch_idx % 50 == 0 or batch_idx == len(train_dataloader) - 1)
            ):
                progress = (batch_idx + 1) / len(train_dataloader)
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = "█" * filled_length + "-" * (bar_length - filled_length)
                print(
                    f"\rTask {task_id + 1:1}, Epoch {epoch+1:>2}/{NUM_EPOCHS}: |{bar}| {progress:.1%} (Batch {batch_idx + 1}/{len(train_dataloader)})",
                    end="",
                    flush=True,
                )

        # Print newline after progress bar completion
        if not QUICK_TEST_MODE and VERBOSE:
            print()

        # Track epoch loss
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        task_epoch_losses.append(avg_epoch_loss)

        # Evaluate performance after each epoch
        t.start("eval")
        model.eval()
        avg_accuracy = accuracy_test.evaluate_tasks_up_to(
            model, criterion, test_dataloaders, task_id, device=device
        )

        # Evaluate on individual tasks for detailed tracking
        individual_accuracies = []
        for eval_task_id in range(task_id + 1):
            eval_dataloader = test_dataloaders[eval_task_id]
            task_acc = accuracy_test.evaluate_single_task(
                model, criterion, eval_dataloader, device=device
            )
            individual_accuracies.append(task_acc)
        t.end("eval")

        # Update visualizer with epoch metrics
        memory_size = clustering_memory.get_memory_size()
        current_lr = (
            lr_scheduler.get_lr() if USE_LEARNING_RATE_SCHEDULER else LEARNING_RATE
        )

        visualizer.update_metrics(
            task_id=task_id,
            overall_accuracy=avg_accuracy,
            individual_accuracies=individual_accuracies,
            epoch_losses=[avg_epoch_loss],
            memory_size=memory_size,
            training_time=None,
            learning_rate=current_lr,
        )

        # Print epoch summary
        if QUICK_TEST_MODE and (epoch % 5 == 0 or epoch == NUM_EPOCHS - 1):
            print(
                f"  Epoch {epoch+1}/{NUM_EPOCHS}: Loss = {avg_epoch_loss:.4f}, Accuracy = {avg_accuracy:.4f}"
            )

    # Calculate training time for this task
    task_time = time.time() - task_start_time

    # Final task summary
    pool_sizes = clustering_memory.get_pool_sizes()
    num_active_pools = clustering_memory.get_num_active_pools()
    final_memory_size = clustering_memory.get_memory_size()

    # Get final accuracy from last epoch evaluation
    final_avg_accuracy = (
        visualizer.task_accuracies[-1] if visualizer.task_accuracies else 0.0
    )

    print(f"After Task {task_id + 1}, Final Average Accuracy: {final_avg_accuracy:.4f}")
    print(
        f"Memory Size: {final_memory_size} samples across {num_active_pools} active pools"
    )
    print(f"Pool sizes: {pool_sizes}")
    print(f"Task Training Time: {task_time:.2f}s")

t.end("training")
print("\nTraining complete.")

# Calculate and display final average accuracy over all epochs and tasks
if visualizer.epoch_data:
    total_accuracy = sum(ep["overall_accuracy"] for ep in visualizer.epoch_data)
    total_epochs = len(visualizer.epoch_data)
    final_average_accuracy = total_accuracy / total_epochs
    print(
        f"\nFinal Average Accuracy (all epochs, all tasks): {final_average_accuracy:.4f}"
    )

# --- 3. Comprehensive Visualization and Analysis ---
print("\nGenerating comprehensive analysis...")

# Save metrics for future analysis
timestamp = time.strftime("%Y%m%d_%H%M%S")

# Build filename with task type, quick test mode, and dataset
task_type_abbrev = {
    "class_incremental": "cla",
    "rotation": "rot",
    "permutation": "perm",
}.get(config["task_type"], config["task_type"][:3])

quick_mode = "q-" if QUICK_TEST_MODE else ""
dataset_name = config["dataset_name"].lower()
filename = f"results-{quick_mode}{task_type_abbrev}-{dataset_name}-{timestamp}.pkl"

visualizer.save_metrics(
    os.path.join(params["output_dir"], filename),
    params=params,
)

# Generate simplified report with 3 key visualizations
visualizer.generate_simple_report(
    clustering_memory,
    show_images=config["show_images"],
)

print(f"\nAnalysis complete! Files saved with timestamp: {timestamp}")
print(t)
