# This is the file pulling it all together. Edit sparingly, if at all!
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
    test_dataloaders,
    visualizer,
) = initialize_system()

# Extract commonly used variables from config
QUICK_TEST_MODE = config["lite"]
VERBOSE = config["verbose"]
# NUM_EPOCHS = config["num_epochs"] # Not used
USE_LEARNING_RATE_SCHEDULER = config["use_learning_rate_scheduler"]
LEARNING_RATE = config["learning_rate"]
BATCH_SIZE = config["batch_size"]
SAMPLING_RATE = config["sampling_rate"]
TESTING_RATE = config["testing_rate"]
t.start("training")

# --- 2. Training Loop ---
print("Starting training...")
print(
    f"""
Quick Test mode: {QUICK_TEST_MODE} | Task Type: {config['task_type']}
Random EM sampling: {config["random_em"]} | Dataset: {config['dataset_name']}
Use LR: {USE_LEARNING_RATE_SCHEDULER} | Sampling Rate: {SAMPLING_RATE}
Only One Epoch: {config["only_one_epoch"]} | Total tasks: {len(train_dataloaders)}"""
)

for task_id, train_dataloader in enumerate(train_dataloaders):
    print(f"\n--- Training on Task {task_id + 1} ---")
    task_start_time = time.time()
    samples_added = 0
    samples_seen = 0
    # Loss accumulation for averaging between testing intervals
    accumulated_loss = 0.0
    loss_count = 0
    avg_accuracy = 0.0  # Initialize to avoid undefined reference

    # Used for debug only
    batches_per_task = len(train_dataloader)
    milestone_25 = batches_per_task // 4
    milestone_50 = batches_per_task // 2
    milestone_75 = (3 * batches_per_task) // 4

    for batch_idx, (data, labels) in enumerate(train_dataloader):
        model.train()
        # Step 1: Update the clustered memory with some samples from this task's dataloader
        # This is where the core clustering for TA-A-GEM happens
        t.start("add samples")
        samples_seen += len(data)
        # Add samples per batch based on sampling_rate
        if SAMPLING_RATE < 1:
            # Fractional sampling - add every 1/SAMPLING_RATE batches
            if batch_idx % int(1 / SAMPLING_RATE) == 0:
                samples_added += 1
                clustering_memory.add_sample(data[0].cpu(), labels[0].cpu(), task_id)
                # Track oldest task IDs after adding sample
                visualizer.track_oldest_task_ids(clustering_memory, task_id)
        else:
            # Sample multiple items per batch (up to batch size and sampling rate)
            num_to_sample = min(int(SAMPLING_RATE), len(data))
            for i in range(num_to_sample):
                samples_added += 1
                clustering_memory.add_sample(data[i].cpu(), labels[i].cpu(), task_id)
                # Track oldest task IDs after adding sample
                visualizer.track_oldest_task_ids(clustering_memory, task_id)
        t.end("add samples")

        t.start("get samples")
        # Move data to device
        data, labels = data.to(device), labels.to(device)
        # Select memory samples for this batch
        if config["random_em"]:
            batch_samples = clustering_memory.get_random_samples(BATCH_SIZE)
        else:
            batch_samples = clustering_memory.get_memory_samples(BATCH_SIZE)
        t.end("get samples")

        # Step 2: Use A-GEM logic for current batch and current memory
        # agem_handler.optimize handles model update and gradient projection
        t.start("optimize")
        batch_loss = agem_handler.optimize(data, labels, batch_samples)
        t.end("optimize")

        # Track batch loss
        if batch_loss is not None:
            # Accumulate loss for averaging
            accumulated_loss += batch_loss
            loss_count += 1
            visualizer.add_batch_loss(task_id, None, batch_idx, batch_loss)

        # Progress bar and milestone info
        if (batch_idx + 1) % 50 == 0 or batch_idx + 1 in [
            milestone_25,
            milestone_50,
            milestone_75,
            batches_per_task,
        ]:
            quarter_size = batches_per_task // 4
            quarter_num = min(4, (batch_idx // quarter_size) + 1)
            quarter_start = (quarter_num - 1) * quarter_size
            quarter_end = (
                quarter_num * quarter_size if quarter_num < 4 else batches_per_task
            )
            quarter_progress = (
                ((batch_idx + 1) - quarter_start) / (quarter_end - quarter_start) * 100
            )
            bar_length = 50
            filled_length = int(bar_length * quarter_progress / 100)
            bar = "█" * filled_length + "-" * (bar_length - filled_length)
            print(
                f"\r{bar} {quarter_progress:.1f}% done Q{quarter_num}. Batch {batch_idx-quarter_start+1} of {quarter_size}",
                end="",
                flush=True,
            )

            # Milestone info at 25, 50, 75%, and end of task
            if batch_idx + 1 in [milestone_25, milestone_50, milestone_75, batches_per_task]:
                # Complete the current quarter's progress bar to 100% before milestone
                if batch_idx + 1 != batches_per_task:  # Don't repeat for final batch
                    completed_bar = "█" * bar_length
                    print(f"\r{completed_bar} 100.0% done Q{quarter_num}. Quarter complete!", end="\n", flush=True)

                print(
                    f"{batch_idx+1} batches done of this task, Accuracy = {avg_accuracy:.4f}"
                )
                print(
                    f"  Added {samples_added} out of {samples_seen} samples to memory."
                )
                print(
                    f"  Sample throughput (cumulative): {clustering_memory.get_sample_throughputs()})"
                )
                samples_added = 0
                samples_seen = 0

        # Evaluate performance after every testing_rate batches
        if (batch_idx + 1) % TESTING_RATE == 0:
            t.start("eval")
            t.start("eval overall")
            model.eval()
            avg_accuracy = accuracy_test.evaluate_tasks_up_to(
                model, criterion, test_dataloaders, task_id, device=device
            )
            t.end("eval overall")

            # Evaluate on individual tasks for detailed tracking
            t.start("eval individual")
            individual_accuracies = []
            for eval_task_id in range(task_id + 1):
                eval_dataloader = test_dataloaders[eval_task_id]
                task_acc = accuracy_test.evaluate_single_task(
                    model, criterion, eval_dataloader, device=device
                )
                individual_accuracies.append(task_acc)
            t.end("eval individual")

            t.end("eval")
            t.start("visualizer")
            # Update visualizer with epoch metrics
            memory_size = clustering_memory.get_memory_size()
            current_lr = (
                lr_scheduler.get_lr() if USE_LEARNING_RATE_SCHEDULER else LEARNING_RATE
            )

            # Calculate average loss since last testing interval
            avg_loss = accumulated_loss / loss_count if loss_count > 0 else 0.0

            visualizer.update_metrics(
                task_id=task_id,
                overall_accuracy=avg_accuracy,
                individual_accuracies=individual_accuracies,
                # Fake "epoch" loss with testing interval loss.
                epoch_loss=avg_loss,
                memory_size=memory_size,
                training_time=None,
                learning_rate=current_lr,
            )
            # Reset loss accumulation
            accumulated_loss = 0.0
            loss_count = 0
            t.end("visualizer")

    # Calculate training time for this task
    task_time = time.time() - task_start_time

    # Final task summary - debug
    pool_sizes = clustering_memory.get_pool_sizes()
    num_active_pools = clustering_memory.get_num_active_pools()
    final_memory_size = clustering_memory.get_memory_size()

    # Get final accuracy from last epoch evaluation
    final_avg_accuracy = (
        visualizer.task_accuracies[-1] if visualizer.task_accuracies else 0.0
    )

    print(f"\nFor task {task_id + 1}, final accuracy was: {final_avg_accuracy:.4f}")
    print(
        f"Memory size: {final_memory_size} samples across {num_active_pools} active pools"
    )
    print(f"Pool sizes: {pool_sizes}")
    print(f"Task training time: {task_time:.2f}s")

t.end("training")
print("\nTraining complete.")

# Calculate and display final average accuracy over all batches and tasks
if visualizer.epoch_data:
    total_accuracy = sum(ep["overall_accuracy"] for ep in visualizer.epoch_data)
    total_batches = len(visualizer.epoch_data)
    final_average_accuracy = total_accuracy / total_batches
    print(
        f"\nFinal Average Accuracy (all batches, all tasks): {final_average_accuracy:.4f}"
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
one_epoch = "e1-" if config["only_one_epoch"] else ""
filename = f"results-{quick_mode}{random_em}{one_epoch}{SAMPLING_RATE}-{task_type_abbrev}-{dataset_name}-{timestamp}.pkl"

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
