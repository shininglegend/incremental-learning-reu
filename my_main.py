# Putting things here in case Titus' new main.py messes up my shit

# This is the file pulling it all together. Edit sparingly, if at all!
import math
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
    epoch_order,
    epochs_visited,
    task_times,
    tasks_seen,
) = initialize_system()

# Extract commonly used variables from config
QUICK_TEST_MODE = config["lite"]
VERBOSE = config["verbose"]
NUM_EPOCHS = config["num_epochs"][0] # I changed this to a dictionary.
USE_LEARNING_RATE_SCHEDULER = config["use_learning_rate_scheduler"]
LEARNING_RATE = config["learning_rate"]

t.start("training")

# --- 2. Training Loop ---
print("Starting training...")
print(
    f"""
Quick Test mode: {QUICK_TEST_MODE} | Task Type: {config['task_type']}
Removal method: {config.get("removal")}  |  Consider newest: {config.get('consider_newest')}
Total tasks: {len(train_dataloaders)}  |  Task introduction style: {config['task_introduction']}"""

)

# For a print statement in the main loop, lowk unnecessary but I like it
check_tasks_seen = True

for epoch, task_id in enumerate(epoch_order):
    epochs_visited[task_id] += 1  # We've seen one more epoch of this task
    if task_id not in tasks_seen:
        tasks_seen.append(task_id)

    print(f"\n--- Training on Task {task_id + 1}, Epoch {epochs_visited[task_id]} ---")

    epoch_start_time = time.time()
    task_epoch_losses = []


    model.train()
    epoch_loss = 0.0
    num_batches = 0

    train_dataloader = train_dataloaders[task_id]

    for batch_idx, (data, labels) in enumerate(train_dataloader):
        # Move data to device
        data, labels = data.to(device), labels.to(device)


        # Step 1: Use A-GEM logic for current batch with clustering memory
        t.start("optimize")
        batch_loss = agem_handler.optimize(data, labels, clustering_memory)
        t.end("optimize")


        # Track batch loss
        if batch_loss is not None:
            epoch_loss += batch_loss
            visualizer.add_batch_loss(task_id, epoch, batch_idx, batch_loss)

        t.start('add samples')
        # Step 2: Update the clustered memory with current batch samples
        sample_data = data[0].to(device)
        sample_label = labels[0].to(device)
        clustering_memory.update_memory(
            sample_data, sample_label, task_id
        )  # Add sample to clusters
        t.end("add samples")


        num_batches += 1

        # Update progress bar every 10 batches or on last batch
        if (
                not QUICK_TEST_MODE
                and VERBOSE
                and ((batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_dataloader))
        ):
            progress = (batch_idx + 1) / len(train_dataloader)
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = "█" * filled_length + "-" * (bar_length - filled_length)
            print(
                f"\rTask {task_id + 1:1}, Epoch {epochs_visited[task_id]}/{NUM_EPOCHS}: |{bar}| {progress:.1%} (Batch {batch_idx + 1}/{len(train_dataloader)})",
                end="",
                flush=True,
            )

        # Print newline after progress bar completion
    if not QUICK_TEST_MODE and VERBOSE:
        print()

        # Track epoch loss
    avg_epoch_loss = epoch_loss / max(num_batches, 1)
    task_epoch_losses.append(avg_epoch_loss)

    # Evaluation code
    t.start("eval")
    model.eval()

    # Evaluate on individual tasks for detailed tracking
    individual_accuracies = []
    for _, eval_task_id in enumerate(tasks_seen):
        eval_dataloader = test_dataloaders[eval_task_id]
        task_acc = accuracy_test.evaluate_single_task(
            model, criterion, eval_dataloader, device=device
        )
        individual_accuracies.append(task_acc)

    avg_accuracy = sum(individual_accuracies) / len(individual_accuracies)

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
        epoch_loss=[avg_epoch_loss],
        memory_size=memory_size,
        training_time=None,
        learning_rate=current_lr,
    )

    # Calculate training time for this epoch
    epoch_time = time.time() - epoch_start_time
    task_times[task_id] += epoch_time

    # Print epoch summary
    if QUICK_TEST_MODE:
        print(
            f"  Epoch {epoch}: Loss = {avg_epoch_loss:.4f}, Accuracy = {avg_accuracy:.4f}"
        )
    else:
        print(f"Epoch accuracy: {avg_accuracy:.4f}, Epoch loss: {avg_epoch_loss:4f}")

    # Every 10 epochs, print a summary of memory and accuracy.
    if (epoch + 1) % 10 == 0:
        pool_sizes = clustering_memory.get_pool_sizes()
        num_active_pools = clustering_memory.get_num_active_pools()
        final_memory_size = clustering_memory.get_memory_size()

        print("Hey baddie, it's been 10 epochs. Let's see what's going on:")
        print(
            f"Memory Size: {final_memory_size} samples across {num_active_pools} active pools"
        )
        print(f"Pool sizes: {pool_sizes}")

        if check_tasks_seen and (len(tasks_seen) < config['num_tasks']):
            print(f"Tasks seen: {sorted(tasks_seen)}")
        else:
            if check_tasks_seen:
                print("hey girl, we've seen all the tasks by now. Just fyi.")
                print(f"It took {epoch + 1} epochs to get here. Funny how time flies, right?\n")
            check_tasks_seen = False

    if epochs_visited[task_id] == config['num_epochs'][task_id]:
        # The task is finished! Let's print a summary:
        print(f"Task {task_id + 1} completed! We spent {task_times[task_id]:.2f}s training this task.")
        final_avg_accuracy = (visualizer.task_accuracies[-1] if visualizer.task_accuracies else 0.0)
        print(f"After Task {task_id + 1}, Final Average Accuracy: {final_avg_accuracy:.4f}")

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
random_em = "rem-" if config["random_em"] else ""
dataset_name = config["dataset_name"].lower()
filename = (
    f"results-{quick_mode}{random_em}{task_type_abbrev}-{dataset_name}-{timestamp}.pkl"
)

visualizer.save_metrics(
    os.path.join(params["output_dir"], filename),
    params=config,
)

# Generate simplified report with 3 key visualizations
visualizer.generate_simple_report(
    clustering_memory,
    # clusters_to_show=clustering_memory.num_pools,
    show_images=config["show_images"],
)

print(f"\nAnalysis complete! Files saved with timestamp: {timestamp}")
print(t)

# End MLFlow run
visualizer.end_run()
