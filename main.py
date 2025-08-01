# This is the file pulling it all together. Edit sparingly, if at all!
import math
import time, os
from utils import accuracy_test
from utils.task_introduction import MixedTaskDataLoader
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
    _, # Clustering memory is ignored.
    bgd_handler, # Replaces agem_handler
    train_dataloaders,
    test_dataloaders,  # list for legacy compatibility
    visualizer,
    epoch_list,
    train_dataloaders_dict,
    num_epochs_per_task,  # dict of task id to number of epochs it'll run
) = initialize_system()

# Extract commonly used variables from config
QUICK_TEST_MODE = config["lite"]
VERBOSE = config["verbose"]
TOTAL_EPOCHS = config["num_epochs"] * config["num_tasks"]
USE_LEARNING_RATE_SCHEDULER = config["use_learning_rate_scheduler"]
LEARNING_RATE = config["learning_rate"]
BATCH_SIZE = config["batch_size"]
SAMPLING_RATE = config["sampling_rate"]
CONTINUAL_LEARNING = config["task_introduction"] == "continuous"

tasks_seen = []  # Used so we're only testing on tasks the model has seen before.
epochs_seen_per_task = []  # So we can track how many epochs per task we've seen before.
task_training_times = (
    []
)  # So we can track task training times even when the epochs are separated.
task_losses = []
memory_snapshot = []
samples_seen = 0

# Initializations for lists who have indices directly corresponding to task_id
for i in range(config["num_tasks"]):
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
Total tasks: {len(train_dataloaders)}"""
)

for task_id, train_dataloader in enumerate(train_dataloaders):
    print(f"\n--- Training on Task {task_id + 1} ---")

    task_start_time = time.time()
    task_epoch_losses = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (data, labels) in enumerate(train_dataloader):
            # Move data to device
            data, labels = data.to(device), labels.to(device)

            # Step 1: Use A-GEM logic for current batch and current memory
            # agem_handler.optimize handles model update and gradient projection
            t.start("optimize")
            batch_loss = bgd_handler.optimize(data, labels)
            t.end("optimize")

        # Track batch loss
        if batch_loss is not None:
            epoch_loss += batch_loss
            visualizer.add_batch_loss(
                current_task_id,
                (epochs_seen_per_task[current_task_id] - 1),
                batch_idx,
                batch_loss,
            )

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
        t.start("visualizer")
        # Update visualizer with epoch metrics
        current_lr = (
            lr_scheduler.get_lr() if USE_LEARNING_RATE_SCHEDULER else LEARNING_RATE
        )

        visualizer.update_metrics(
            task_id=task_id,
            overall_accuracy=avg_accuracy,
            individual_accuracies=individual_accuracies,
            epoch_loss=avg_epoch_loss,
            memory_size=None,
            training_time=None,
            learning_rate=current_lr,
        )
        t.end("visualizer")

        # Print epoch summary
        if QUICK_TEST_MODE and (epoch % 5 == 0 or epoch == NUM_EPOCHS - 1):
            print(
                f"  Epoch {epoch+1}/{NUM_EPOCHS}: Loss = {avg_epoch_loss:.4f}, Accuracy = {avg_accuracy:.4f}"
            )

    # Calculate training time for this task
    task_time = time.time() - task_start_time

        # Get final accuracy from last epoch evaluation
        final_avg_accuracy = (
            visualizer.task_accuracies[-1] if visualizer.task_accuracies else 0.0
        )

    print(
        f"For task {task_id + 1}, final average accuracy landed at: {final_avg_accuracy:.4f}"
    )
    print(f"Task training time: {task_time:.2f}s")

t.end("training")
print("\nTraining complete.")


"""
------------------------------
Visualization and end-of-program code
------------------------------
"""

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

# Build filename with task type, intro type, random em, quick test mode, and dataset
task_type_abbrev = {
    "class_incremental": "cla",
    "rotation": "rot",
    "permutation": "perm",
}.get(config["task_type"], config["task_type"][:3])

task_introduction_abbrev = {
    "sequential": "seq",
    "half and half": "half",
    "random": "rand",
    "continuous": "cont",
}.get(config["task_introduction"], config["task_introduction"][:3])

quick_mode = "q-" if QUICK_TEST_MODE else ""
random_em = "rem-" if config["random_em"] else ""
dataset_name = config["dataset_name"].lower()
filename = f"results-{quick_mode}{task_introduction_abbrev}-{random_em}{SAMPLING_RATE}-{task_type_abbrev}-{dataset_name}-{timestamp}.pkl"

visualizer.save_metrics(
    os.path.join(params["output_dir"], filename),
    params=params,
)

# Generate simplified report with 3 key visualizations
visualizer.generate_simple_report(
    None,
    show_images=config["show_images"],
)

print(f"\nAnalysis complete! Files saved with timestamp: {timestamp}")
print(t)

# End MLFlow run
visualizer.end_run()
