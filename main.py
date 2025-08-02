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
    bgd_handler,
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

    if CONTINUAL_LEARNING and next_task_id >= config["num_tasks"]:
        # Loops back to the first task.
        # Needs to be disabled in both the dataloader and here if you want it turned off
        next_task_id = -1

    if epochs_seen_per_task[current_task_id] == num_epochs_per_task[current_task_id]:
        end_of_task = True

    # Tracks when we've trained on all tasks for at least one epoch.
    if (next_task_id not in tasks_seen) and (next_task_id != -1):
        tasks_seen.append(next_task_id)
        if len(tasks_seen) == config["num_tasks"]:
            print("\nAll tasks have been seen at least once.")

    # Train the model per batch.
    model.train()

    """Per batch training loop"""
    for batch_idx, (data, labels) in enumerate(epoch_dataloader):
        # Move data to device
        data, labels = data.to(device), labels.to(device)

        # optimize handles model update
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
            and (batch_idx % 50 == 0 or batch_idx == len(epoch_dataloader) - 1)
        ):
            progress = (batch_idx + 1) / len(epoch_dataloader)
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = "█" * filled_length + "-" * (bar_length - filled_length)
            task_str = (
                f"Tasks {current_task_id + 1:1} and {next_task_id + 1:1}"
                if current_task_id != next_task_id
                else f"Task {current_task_id + 1}"
            )
            print(
                f"\r{task_str}, Epoch {epochs_seen_per_task[current_task_id]:>2}/"
                f"{num_epochs_per_task[current_task_id]}:"
                f" |{bar}| {progress:.1%} (Batch {batch_idx + 1}/{len(epoch_dataloader)})",
                end="",
                flush=True,
            )

    epoch_training_time = time.time() - epoch_start_time
    task_training_times[current_task_id] += epoch_training_time

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

    t.start("visualizer")
    # Update visualizer with epoch metrics
    current_lr = lr_scheduler.get_lr() if USE_LEARNING_RATE_SCHEDULER else LEARNING_RATE

    visualizer.update_metrics(
        task_id=current_task_id,
        overall_accuracy=avg_accuracy,
        individual_accuracies=individual_accuracies,
        epoch_loss=avg_epoch_loss,
        memory_size=None,
        training_time=time.time() - epoch_start_time,
        learning_rate=current_lr,
    )
    t.end("visualizer")

    # Print epoch summary
    if QUICK_TEST_MODE and (
        epoch_number % 5 == 0
        or epoch_number == num_epochs_per_task[current_task_id] - 1
    ):
        print(
            f"  Epoch {epoch_number + 1}/{num_epochs_per_task[current_task_id]}: Loss = {avg_epoch_loss:.4f}, Accuracy = {avg_accuracy:.4f}"
        )

    if not QUICK_TEST_MODE:
        task_str = (
            f"Tasks {current_task_id + 1:1} and {next_task_id + 1:1}"
            if current_task_id != next_task_id
            else f"Task {current_task_id + 1}"
        )
        final_output_str = (
            f"\r{task_str}, Epoch {epochs_seen_per_task[current_task_id]:>2}/{num_epochs_per_task[current_task_id]}:"
            f" Accuracy: {avg_accuracy:.2%} Loss {avg_epoch_loss:.4}"
        )

        # Print the final string, padding with spaces, and then a newline
        print(f"{final_output_str:<80}")  # Adjust padding width as needed

    # End of task summary
    if end_of_task:
        # Calculate training time for this task
        task_time = task_training_times[current_task_id]

        # Get final accuracy from last epoch evaluation
        final_avg_accuracy = (
            visualizer.task_accuracies[-1] if visualizer.task_accuracies else 0.0
        )

        print(
            f"After fully training task {current_task_id + 1}, average accuracy landed at: {final_avg_accuracy:.4f}"
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
    show_images=config["show_images"],
)

print(f"\nAnalysis complete! Files saved with timestamp: {timestamp}")
print(t)

# End MLFlow run
visualizer.end_run()
