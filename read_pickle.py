#!./env/bin/python
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Debug and display controls
DEBUG_MODE = False
INDIVIDUAL_PLOTS_TO_DISPLAY = {
    "dashboard": False,
    # Accuracy
    "per_task_accuracies": False,  # Matrix of current task to accuracy on previous task
    "task_accuracies": False,  # Line graph
    # Loss
    "batch_losses": False,
    "epoch_losses": False,
    # Other
    "training_times": False,
    "memory_efficiency": False,
    "oldest_task_ids_tracking": True,  # Bar chart of oldest task IDs over time
}

# If you want to save the plots, change this path. Helpful for linux. Will save all files.
SAVE_DIR = None
if SAVE_DIR and not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

filename = input(
    "Enter the filename to load (e.g., 'ta_agem_metrics_2025-07-01T15:45:44.660534.pkl'): "
)

# Look for it in the test_results directory, if not found, use the relative path
if os.path.exists(os.path.join("test_results", filename)):
    filename = os.path.join("test_results", filename)

with open(filename, "rb") as file:
    data = pickle.load(file)

    # Convert new format to legacy format if needed
    if "epoch_data" in data and data["epoch_data"]:
        print(
            "Detected new epoch_data format, converting to legacy format for visualization..."
        )
        epoch_data = data["epoch_data"]

        # Convert to legacy format
        data["task_accuracies"] = [ep["overall_accuracy"] for ep in epoch_data]
        data["per_task_accuracies"] = [ep["individual_accuracies"] for ep in epoch_data]
        data["memory_sizes"] = [ep["memory_size"] for ep in epoch_data]
        data["epoch_losses"] = [[ep["epoch_loss"]] for ep in epoch_data]
        data["memory_efficiency"] = [
            ep["overall_accuracy"] / ep["memory_size"] if ep["memory_size"] > 0 else 0.0
            for ep in epoch_data
        ]

        # Group by task for epoch losses
        task_epoch_losses = {}
        for ep in epoch_data:
            task_id = ep["task_id"]
            if task_id not in task_epoch_losses:
                task_epoch_losses[task_id] = []
            task_epoch_losses[task_id].append(ep["epoch_loss"])

        data["epoch_losses"] = [
            task_epoch_losses[i] for i in sorted(task_epoch_losses.keys())
        ]

    # See what the loaded data looks like without printing the entire structure
    if isinstance(data, dict):
        print(f"Loaded data contains {len(data)} keys:")
        for key in data.keys():
            print(f" - {key}: {type(data[key])}")
        print()
        if data["params"]:
            print("-- Paramaters --")
            for param_key in data["params"].keys():
                print(f"{param_key}: {data['params'].get(param_key, None)}")
            print()
    elif isinstance(data, list):
        print(f"Loaded data is a list with {len(data)} items.")
        if len(data) > 0:
            print(f" - First item type: {type(data[0])}")
    else:
        print(
            f"Loaded data is of type {type(data)} with length {len(data) if hasattr(data, '__len__') else 'N/A'}"
        )

# Print the first few entries of each key to inspect the data
if DEBUG_MODE:
    for key in data.keys():
        if isinstance(data[key], list):
            if len(data[key]) > 0 and isinstance(data[key][0], list):
                print(
                    f"{key}: {len(data[key])} items, first item: {data[key][0][:3] if len(data[key][0]) > 3 else data[key][0]}"
                )
            else:
                print(f"{key}: {data[key][:5]}")  # Print first 5 items for lists
        else:
            print(f"{key}: {data[key]}")  # Print the value directly for non-list types


def show_or_open(figure: go.Figure, type):
    figure.update_layout(title_subtitle_text=filename)
    if SAVE_DIR:
        figure.write_html(os.path.join(SAVE_DIR, f"{type}.html"))
    if INDIVIDUAL_PLOTS_TO_DISPLAY[type]:
        figure.show()


# Create visualizations for accuracy and loss data

# 1. Task Accuracies Over Time (Individual Task Performance)
if "per_task_accuracies" in data and data["per_task_accuracies"]:
    fig_task_acc = go.Figure()

    # Create a line for each task showing its accuracy over time
    for task_id in range(len(data["per_task_accuracies"])):
        # Extract accuracy for this specific task across all training stages
        task_performance = []
        x_values = []

        # For each training stage, get the accuracy of this specific task
        for stage, stage_accuracies in enumerate(data["per_task_accuracies"]):
            if task_id < len(stage_accuracies):  # Only if this task has been evaluated
                task_performance.append(stage_accuracies[task_id])
                x_values.append(stage + 1)  # Task numbers start from 1

        # Add line for this task
        fig_task_acc.add_trace(
            go.Scatter(
                x=x_values,
                y=task_performance,
                mode="lines+markers",
                name=f"Task {task_id + 1}",
                line=dict(width=2.5),
                marker=dict(size=6),
            )
        )

    fig_task_acc.update_layout(
        title="Individual Task Accuracies Over Time",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1]),
        legend_title="Task",
        hovermode="x unified",
    )
    show_or_open(fig_task_acc, "task_accuracies")

# 2. Per-Task Accuracies (Matrix/Heatmap)
if "per_task_accuracies" in data and data["per_task_accuracies"]:
    # Create a matrix showing accuracy for each task at each epoch
    max_tasks = (
        max(len(epoch_accs) for epoch_accs in data["per_task_accuracies"])
        if data["per_task_accuracies"]
        else 0
    )
    accuracy_matrix = []

    for i, epoch_accs in enumerate(data["per_task_accuracies"]):
        row = epoch_accs + [None] * (max_tasks - len(epoch_accs))
        accuracy_matrix.append(row)

    fig_per_task = px.imshow(
        accuracy_matrix,
        title="Per-Task Accuracies Over Epochs",
        labels={"x": "Task", "y": "Epoch", "color": "Accuracy"},
        aspect="auto",
        color_continuous_scale="Viridis",
    )
    fig_per_task.update_layout(xaxis_title="Task", yaxis_title="Epoch")
    show_or_open(fig_per_task, "per_task_accuracies")

# 3. Epoch Losses for Each Task
if "epoch_losses" in data and data["epoch_losses"]:
    fig_epoch_losses = go.Figure()

    for i, task_losses in enumerate(data["epoch_losses"]):
        fig_epoch_losses.add_trace(
            go.Scatter(
                x=list(range(1, len(task_losses) + 1)),
                y=task_losses,
                mode="lines+markers",
                name=f"Task {i+1}",
                line=dict(width=2),
            )
        )

    fig_epoch_losses.update_layout(
        title="Epoch Losses for Each Task",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend_title="Task",
    )
    show_or_open(fig_epoch_losses, "epoch_losses")

# 4. Batch Losses Over Time (if available)
if "batch_losses" in data and data["batch_losses"]:
    # Create a dataframe for batch losses
    batch_df = pd.DataFrame(data["batch_losses"])

    if not batch_df.empty:
        fig_batch_losses = px.line(
            batch_df,
            x=batch_df.index,
            y="loss",
            color="task",
            title="Batch Losses Over Time",
            labels={"index": "Batch Number", "loss": "Loss", "task": "Task"},
        )
        fig_batch_losses.update_layout(xaxis_title="Batch Number", yaxis_title="Loss")
        show_or_open(fig_batch_losses, "batch_losses")

# 5. Training Times Per Task
if "training_times" in data and data["training_times"]:
    fig_training_times = px.bar(
        x=list(range(1, len(data["training_times"]) + 1)),
        y=data["training_times"],
        title="Training Times Per Task",
        labels={"x": "Task Number", "y": "Training Time (seconds)"},
    )
    fig_training_times.update_layout(
        xaxis_title="Task Number", yaxis_title="Training Time (seconds)"
    )
    show_or_open(fig_training_times, "training_times")

# 6. Memory Efficiency Over Tasks
if "memory_efficiency" in data and data["memory_efficiency"]:
    fig_memory_eff = px.line(
        x=list(range(1, len(data["memory_efficiency"]) + 1)),
        y=data["memory_efficiency"],
        title="Memory Efficiency Over Tasks",
        labels={"x": "Task Number", "y": "Memory Efficiency"},
        markers=True,
    )
    fig_memory_eff.update_layout(
        xaxis_title="Task Number", yaxis_title="Memory Efficiency"
    )
    show_or_open(fig_memory_eff, "memory_efficiency")

# 7. Combined Accuracy and Loss Dashboard
if (
    "task_accuracies" in data
    and "epoch_losses" in data
    and data["task_accuracies"]
    and data["epoch_losses"]
):
    # Create subplots
    fig_combined = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Overall Accuracy per Epoch",
            "Average Epoch Loss per Task",
            "Individual Task Performance",
            "Forgetting per Task",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Task accuracies (overall accuracy per epoch)
    fig_combined.add_trace(
        go.Scatter(
            x=list(range(1, len(data["task_accuracies"]) + 1)),
            y=data["task_accuracies"],
            mode="lines+markers",
            name="Overall Accuracy",
            line=dict(color="blue", width=3),
        ),
        row=1,
        col=1,
    )

    # Average epoch loss per task
    avg_losses = [
        sum(task_losses) / len(task_losses) for task_losses in data["epoch_losses"]
    ]
    fig_combined.add_trace(
        go.Scatter(
            x=list(range(1, len(avg_losses) + 1)),
            y=avg_losses,
            mode="lines+markers",
            name="Avg Loss",
            line=dict(color="red", width=3),
        ),
        row=1,
        col=2,
    )

    # Per-task accuracies (individual task performance over time)
    for task_id in range(len(data["per_task_accuracies"])):
        # Extract accuracy for this specific task across all training stages
        task_performance = []
        x_values = []

        # For each training stage, get the accuracy of this specific task
        for stage, stage_accuracies in enumerate(data["per_task_accuracies"]):
            if task_id < len(stage_accuracies):  # Only if this task has been evaluated
                task_performance.append(stage_accuracies[task_id])
                x_values.append(stage + 1)  # Task numbers start from 1

        # Add line for this task
        fig_combined.add_trace(
            go.Scatter(
                x=x_values,
                y=task_performance,
                mode="lines+markers",
                name=f"Task {task_id + 1}",
                showlegend=True,
                line=dict(width=2),
            ),
            row=2,
            col=1,
        )

    # Calculate forgetting for each task over epochs
    if "per_task_accuracies" in data and data["per_task_accuracies"]:
        max_tasks = (
            max(len(epoch_accs) for epoch_accs in data["per_task_accuracies"])
            if data["per_task_accuracies"]
            else 0
        )

        for task_id in range(max_tasks):
            task_forgetting = []
            x_values = []
            max_so_far = 0

            # Extract accuracies for this task across all epochs where it's evaluated
            for epoch_idx, epoch_accs in enumerate(data["per_task_accuracies"]):
                if task_id < len(epoch_accs):
                    accuracy = epoch_accs[task_id]
                    if max_so_far == 0:  # First occurrence
                        max_so_far = accuracy
                    else:
                        max_so_far = max(max_so_far, accuracy)

                    # Calculate forgetting: Fl = max({Ai|i ≤ l}) − Al
                    forgetting = max_so_far - accuracy
                    task_forgetting.append(forgetting)
                    x_values.append(epoch_idx + 1)

            if task_forgetting:
                # Add line for this task's forgetting over time
                fig_combined.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=task_forgetting,
                        mode="lines+markers",
                        name=f"Task {task_id + 1} Forgetting",
                        showlegend=True,
                        line=dict(width=2),
                    ),
                    row=2,
                    col=2,
                )
                fig_combined.update_yaxes(range=[0.0, 1.0])

        if DEBUG_MODE:
            print(f"Calculated forgetting over epochs for {max_tasks} tasks")
    else:
        # Add placeholder text when no forgetting data available
        fig_combined.add_annotation(
            x=0.5,
            y=0.5,
            text="No forgetting data available",
            showarrow=False,
            row=2,
            col=2,
            xref="x domain",
            yref="y domain",
            font=dict(size=14, color="gray"),
        )

    fig_combined.update_layout(
        title_text="Incremental Learning Performance Dashboard",
        showlegend=True,
        height=800,
    )

    # Update axis labels
    fig_combined.update_xaxes(title_text="Epoch", row=1, col=1)
    fig_combined.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig_combined.update_xaxes(title_text="Task", row=1, col=2)
    fig_combined.update_yaxes(title_text="Average Loss", row=1, col=2)
    fig_combined.update_xaxes(title_text="Epoch", row=2, col=1)
    fig_combined.update_yaxes(title_text="Accuracy", row=2, col=1)
    fig_combined.update_xaxes(title_text="Epoch", row=2, col=2)
    fig_combined.update_yaxes(title_text="Forgetting", row=2, col=2)

    show_or_open(fig_combined, "dashboard")

# 8. Oldest Task IDs Tracking Over Time
if "oldest_task_ids_tracking" in data and data["oldest_task_ids_tracking"]:
    # We use matplot lib here cause it works better
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    tracking_data = data["oldest_task_ids_tracking"]

    # Extract matrices and task boundaries
    matrices = []
    task_boundaries = []
    current_task = None

    for i, item in enumerate(tracking_data):
        if isinstance(item, tuple):
            matrix, task_id = item
            matrices.append(matrix)
            if current_task is None:
                current_task = task_id
            elif current_task != task_id:
                task_boundaries.append(i)
                current_task = task_id
        else:
            # Handle old format without task_id
            matrices.append(item)

    # Convert to matrix format for heatmap
    time_steps = len(matrices)
    max_clusters = (
        max(
            len([task for pool_row in matrix for task in pool_row])
            for matrix in matrices
        )
        if matrices
        else 0
    )

    # Create matrix: rows = clusters, columns = time_steps
    heatmap_matrix = np.full((max_clusters, time_steps), np.nan)

    for time_step, matrix in enumerate(matrices):
        cluster_idx = 0
        for pool_row in matrix:
            for task_id in pool_row:
                if cluster_idx < max_clusters:
                    heatmap_matrix[cluster_idx, time_step] = (
                        task_id if task_id is not None else -1
                    )
                cluster_idx += 1

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create custom colormap
    unique_tasks = np.unique(heatmap_matrix[~np.isnan(heatmap_matrix)])
    max_task = (
        int(max(unique_tasks[unique_tasks >= 0]))
        if len(unique_tasks[unique_tasks >= 0]) > 0
        else 0
    )

    # Create custom colormap with white for -1/None
    tab10 = plt.cm.get_cmap("tab10")
    colors = ["white"] + [tab10(i) for i in range(max_task + 1)]
    cmap = ListedColormap(colors)

    # Handle NaN and -1 values
    masked_matrix = np.ma.masked_where(np.isnan(heatmap_matrix), heatmap_matrix)

    im = ax.imshow(masked_matrix, cmap=cmap, aspect="auto", vmin=-1, vmax=max_task)

    ax.set_title("Oldest Task IDs in Clusters Over Time")
    ax.set_ylabel("Cluster Index")

    ax.set_xlabel("Time")
    # Convert x-axis from frame indices to batch numbers if tracking_interval available
    if "tracking_interval" in data["params"]:
        ax.set_xlabel("Batches")
        tracking_interval = data["params"]["tracking_interval"]
        num_ticks = min(10, time_steps)
        tick_positions = [i * time_steps // num_ticks for i in range(num_ticks)]
        tick_labels = [i * tracking_interval for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

    # Add task boundary lines
    for boundary in task_boundaries:
        ax.axvline(
            x=boundary - 0.5, color="black", linestyle="-", linewidth=2, alpha=0.8
        )

    # Create horizontal legend below the plot
    from matplotlib.lines import Line2D

    legend_elements = []

    # Add task entries
    for task_id in range(max_task + 1):
        color = colors[task_id + 1]  # +1 because white is at index 0
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=4, label=f"Task {task_id}")
        )

    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=min(len(legend_elements), 8),
        frameon=False,
    )

    plt.tight_layout()

    if INDIVIDUAL_PLOTS_TO_DISPLAY["oldest_task_ids_tracking"]:
        plt.show()
    if SAVE_DIR:
        plt.savefig(
            os.path.join(SAVE_DIR, "oldest_task_ids_tracking.png"),
            dpi=150,
            bbox_inches="tight",
        )
        print("Saved oldest_task_ids_tracking.png")
    plt.close()

print("\nVisualization complete! All plots have been displayed.")

# Here's whats inside a pickle file
# Loaded data contains 9 keys:
#  - task_accuracies: <class 'list'>
#  - per_task_accuracies: <class 'list'>
#  - task_losses: <class 'list'>
#  - memory_sizes: <class 'list'>
#  - epoch_losses: <class 'list'>
#  - batch_losses: <class 'list'>
#  - training_times: <class 'list'>
#  - memory_efficiency: <class 'list'>
#  - timestamp: <class 'str'>
# task_accuracies: [0.9594333333333334, 0.9546833333333333, 0.9502888888888888, 0.9450708333333333, 0.9070866666666667]
# per_task_accuracies: [[0.9594333333333334], [0.9347, 0.9746666666666667], [0.9268666666666666, 0.9435166666666667, 0.9804833333333334], [0.9209166666666667, 0.9198833333333334, 0.9555, 0.9839833333333333], [0.9032, 0.8334166666666667, 0.8781, 0.9346833333333333, 0.9860333333333333]]
# task_losses: []
# memory_sizes: [300, 300, 300, 300, 300]
# epoch_losses: [[2.0180822248955566, 0.8029772452364365, 0.4473678542605291, 0.36697515671979636, 0.32977847607030225, 0.3046029427026709, 0.28490991294461615, 0.2686583392311974, 0.25382520612771625, 0.2401979106964233, 0.22756280648584168, 0.21587102710541028, 0.2046417263180483, 0.19436636115816266, 0.18492470623898163, 0.17597805076796794, 0.16816842238907703, 0.1605539666544646, 0.15367626194046655, 0.1471087557153854], [0.5414902678420768, 0.30120141042303294, 0.2544803082526972, 0.22620040408933226, 0.20548833864791474, 0.1893829851942525, 0.1758738736019004, 0.1645858362170402, 0.15486780721162602, 0.1463226967815232, 0.13856423099238116, 0.13178723096195608, 0.1257574971980066, 0.1203883143888476, 0.11537995131581556, 0.11050419865457418, 0.10652552801845984, 0.10229626965200683, 0.0987387756666576, 0.09520016171818134], [0.4540941879283637, 0.2512633609777937, 0.20834854503635628, 0.18301143488610008, 0.16517202304082457, 0.15119030676968395, 0.1399812837005399, 0.130712757900008, 0.1231312629658108, 0.11606739094855341, 0.11030774421665895, 0.10484898558687807, 0.10009397633267024, 0.09547206677408152, 0.09176630996819586, 0.08805107915656603, 0.08448509925194472, 0.08139834504655058, 0.07847163966131726, 0.07567049010526292], [0.41401334783722027, 0.2215343707022257, 0.18186534664690648, 0.15839615666780932, 0.1422197986469449, 0.13004636874926898, 0.12032786464349678, 0.11245359925029334, 0.10586371172593984, 0.09987365050616791, 0.09456998741798452, 0.08995312275502754, 0.08569304468423555, 0.08181631202960853, 0.07834924673042648, 0.07514143973880952, 0.07203544019345039, 0.06939931127404755, 0.06679095199821555, 0.06428769577164106], [0.387754821513469, 0.20646680713330473, 0.16732244411289382, 0.1445313779882854, 0.12908334166370333, 0.11749606502095897, 0.10804519921317114, 0.10069201538964989, 0.094081854217938, 0.0888413180408194, 0.08398631283652504, 0.0798929918663974, 0.07567491156457981, 0.07252925403696038, 0.06881062635393755, 0.0661675880897116, 0.06328535220943256, 0.0607973341911226, 0.05833326694372226, 0.05625830742286901]]
# batch_losses: [{'task': 0, 'epoch': 0, 'batch': 0, 'loss': 2.3284037113189697}, {'task': 0, 'epoch': 0, 'batch': 1, 'loss': 2.2965750694274902}, {'task': 0, 'epoch': 0, 'batch': 2, 'loss': 2.332415819168091}, {'task': 0, 'epoch': 0, 'batch': 3, 'loss': 2.325805187225342}, {'task': 0, 'epoch': 0, 'batch': 4, 'loss': 2.2886767387390137}]
# training_times: [664.2958102226257, 707.9371378421783, 710.0332610607147, 713.1663269996643, 949.6074056625366]
# memory_efficiency: [0.003198111111111111, 0.003182277777777778, 0.0031676296296296296, 0.003150236111111111, 0.0030236222222222225]
# timestamp: 2025-07-01T15:45:44.660534
