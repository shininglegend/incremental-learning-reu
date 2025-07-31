# This is the file pulling it all together. Edit sparingly, if at all!
import math
import time, os
import numpy as np
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
    test_dataloaders,  # list for legacy compatibility
    visualizer,
    epoch_list,
    train_dataloaders_dict,
) = initialize_system()

# Extract commonly used variables from config
QUICK_TEST_MODE = config["lite"]
VERBOSE = config["verbose"]
NUM_EPOCHS_PER_TASK = config["num_epochs"]  # With Abby's per-epoch loop, NUM_EPOCHS_PER_TASK is a dictionary.
TOTAL_EPOCHS = config['total_epochs']
USE_LEARNING_RATE_SCHEDULER = config["use_learning_rate_scheduler"]
LEARNING_RATE = config["learning_rate"]
BATCH_SIZE = config["batch_size"]
SAMPLING_RATE = config["sampling_rate"]

tasks_seen = []  # Used so we're only testing on tasks the model has seen before.
epochs_seen_per_task = []   # So we can track how many epochs per task we've seen before.
task_training_times = []  # So we can track task training times even when the epochs are separated.
task_losses = []
memory_snapshot = []

# Initializations for lists who have indices directly corresponding to task_id
for i in range(config['num_tasks']):
    epochs_seen_per_task.append(0)
    task_training_times.append(0)
    task_losses.append(0)


# --- 2. Training Loop ---
t.start("training")
print("Starting training...")
print(
    f"""
Quick Test mode: {QUICK_TEST_MODE} | Task Type: {config['task_type']}
Random EM sampling: {config["random_em"]} | Dataset: {config['dataset_name']}
Use LR: {USE_LEARNING_RATE_SCHEDULER} | Sampling Rate: {SAMPLING_RATE}
Total tasks: {len(train_dataloaders)} | Task introduction type: {config['task_introduction']}"""
)

"""
-------------------------------------------
Main training loop. Per epoch in epoch_list.
-------------------------------------------
"""

for epoch_number, epoch_task_id in enumerate(epoch_list):
    epoch_start_time = time.time()

    epoch_dataloader = train_dataloaders_dict[epoch_task_id]
    end_of_task = False
    epoch_loss = 0
    num_batches = 0

    # Some logic if you're doing continual task introduction.
    current_task_id = math.floor(epoch_task_id)
    next_task_id = math.ceil(epoch_task_id)

    epochs_seen_per_task[current_task_id] += 1

    if current_task_id != next_task_id:
        if next_task_id >= config['num_tasks']:
            next_task_id = -1
            # the next task should be the first one if we're off the end.
        epochs_seen_per_task[next_task_id] += 1

    if epochs_seen_per_task[current_task_id] == NUM_EPOCHS_PER_TASK[current_task_id]:
        end_of_task = True

    # Tracks when we've seen all tasks for at least one epoch.
    # Prints a fun message!
    if (next_task_id not in tasks_seen) and (next_task_id != -1):
        tasks_seen.append(next_task_id)
        if len(tasks_seen) == config['num_tasks']:
            print("\nWith this epoch, we've seen all tasks at least once!")
            print(f"It took {epoch_number + 1} epochs to get here. Funny how time flies, right?")
            print("Anyways, let's get on with the training xoxo.\n")


    # Train the model per batch.
    model.train()

    # Get a fresh copy of memory every 20 epochs.
    if epoch_number % 20 == 0:
        memory_snapshot = clustering_memory.get_memory_samples()

    '''Per batch training loop'''
    for batch_idx, (data, labels) in enumerate(epoch_dataloader):

        # Move data to device
        data, labels = data.to(device), labels.to(device)

        # Select memory samples for this batch
        t.start("sample from memory")
        batch_samples = agem_handler.sample_from_memory(config, memory_snapshot)
        t.end("sample from memory")

        # agem_handler.optimize handles model update and gradient projection
        t.start("optimize")
        batch_loss = agem_handler.optimize(data, labels, batch_samples)
        t.end("optimize")

        # Track batch loss
        if batch_loss is not None:
            epoch_loss += batch_loss
            visualizer.add_batch_loss(current_task_id,
                                      (epochs_seen_per_task[current_task_id] - 1),
                                      batch_idx,
                                      batch_loss)

        num_batches += 1

        # Update progress bar every 50 batches or on last batch
        if (
                not QUICK_TEST_MODE
                and VERBOSE
                and (batch_idx % 50 == 0 or batch_idx == len(epoch_dataloader) - 1)
        ):
            progress = (batch_idx + 1) / len(epoch_dataloader)
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = "█" * filled_length + "-" * (bar_length - filled_length)

            print(
                f"\rTask {epoch_task_id + 1:1}, Epoch {epochs_seen_per_task[current_task_id]:>2}/{NUM_EPOCHS_PER_TASK[epoch_task_id]}:"
                f" |{bar}| {progress:.1%} (Batch {batch_idx + 1}/{len(epoch_dataloader)})",
                end="",
                flush=True,
            )

    epoch_training_time = time.time() - epoch_start_time
    task_training_times[current_task_id] += epoch_training_time

    # Print newline after progress bar completion
    if not QUICK_TEST_MODE and VERBOSE:
        print()

    # Track epoch loss
    avg_epoch_loss = epoch_loss / max(num_batches, 1)

    # Evaluate performance after each epoch
    # <editor-fold desc="Evaluation">
    t.start("eval")
    model.eval()

    avg_accuracy = accuracy_test.evaluate_particular_tasks(
        model, criterion, test_dataloaders, tasks_seen, device=device
    )

    # Evaluate on individual tasks for detailed tracking
    individual_accuracies = []
    for list_idx, eval_task_id in enumerate(tasks_seen):
        eval_dataloader = test_dataloaders[eval_task_id]
        task_acc = accuracy_test.evaluate_single_task(
            model, criterion, eval_dataloader, device=device
        )
        individual_accuracies.append(task_acc)

    t.end("eval")
    # </editor-fold>

    # <editor-fold desc="Visualizer">
    t.start("visualizer")
    # Update visualizer with epoch metrics
    memory_size = clustering_memory.get_memory_size()
    current_lr = (
        lr_scheduler.get_lr() if USE_LEARNING_RATE_SCHEDULER else LEARNING_RATE
    )

    visualizer.update_metrics(
        task_id=current_task_id,
        overall_accuracy=avg_accuracy,
        individual_accuracies=individual_accuracies,
        epoch_loss=avg_epoch_loss,
        memory_size=memory_size,
        training_time=None,
        learning_rate=current_lr,
    )
    t.end("visualizer")
    # </editor-fold>

    # Print epoch summary
    if QUICK_TEST_MODE and (epoch_number % 5 == 0 or epoch_number == NUM_EPOCHS_PER_TASK[epoch_task_id] - 1):
        print(
            f"  Epoch {epoch_number + 1}/{NUM_EPOCHS_PER_TASK[epoch_task_id]}: Loss = {avg_epoch_loss:.4f}, Accuracy = {avg_accuracy:.4f}"
        )


    # Add samples to memory every 20 epochs
    if epoch_number % 20 == 0:
        # Step 2: Update the clustered memory with some samples from this task's dataloader
        # This is where the core clustering for TA-A-GEM happens
        t.start("add samples")
        samples_added = 0
        batch_counter = 0
        # Add samples per batch based on sampling_rate
        for sample_data, sample_labels in epoch_dataloader:
            if SAMPLING_RATE < 1:
                # Fractional sampling - add every 1/SAMPLING_RATE batches
                if batch_counter % int(1 / SAMPLING_RATE) == 0:
                    samples_added += 1
                    clustering_memory.add_sample(
                        sample_data[0].cpu(), sample_labels[0].cpu(), current_task_id
                    )
                    # Track oldest task IDs after adding sample
                    visualizer.track_oldest_task_ids(clustering_memory, current_task_id)
            else:
                # Sample multiple items per batch (up to batch size and sampling rate)
                num_to_sample = min(int(SAMPLING_RATE), len(sample_data))
                for i in range(num_to_sample):
                    samples_added += 1
                    clustering_memory.add_sample(
                        sample_data[i].cpu(), sample_labels[i].cpu(), current_task_id
                    )
                    # Track oldest task IDs after adding sample
                    visualizer.track_oldest_task_ids(clustering_memory, current_task_id)
            batch_counter += 1
        print(
            f"Added {samples_added} out of {len(epoch_dataloader) * BATCH_SIZE} samples this round."
        )
        print("Sample throughput (cumulative):", clustering_memory.get_sample_throughputs())
        t.end("add samples")


    # End of task summary
    if end_of_task:
        # Calculate training time for this task
        task_time = task_training_times[current_task_id]

        # Final task summary
        pool_sizes = clustering_memory.get_pool_sizes()
        num_active_pools = clustering_memory.get_num_active_pools()
        final_memory_size = clustering_memory.get_memory_size()

        # Get final accuracy from last epoch evaluation
        final_avg_accuracy = (
            visualizer.task_accuracies[-1] if visualizer.task_accuracies else 0.0
        )

        print(
            f"After fully training task {current_task_id + 1}, average accuracy landed at: {final_avg_accuracy:.4f}"
        )
        print(
            f"Memory size: {final_memory_size} samples across {num_active_pools} active pools"
        )
        print(f"Pool sizes: {pool_sizes}")
        print(f"Task training time: {task_time:.2f}s")


'''
------------------------------
Visualization and end-of-program code
------------------------------
'''

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
filename = f"results-{quick_mode}{random_em}{SAMPLING_RATE}-{task_type_abbrev}-{dataset_name}-{timestamp}.pkl"

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

# End MLFlow run
visualizer.end_run()
