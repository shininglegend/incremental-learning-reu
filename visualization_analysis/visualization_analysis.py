import pandas as pd
import plotly.express as px
import pickle
import os
from datetime import datetime
import time

# NOTE: Many functions in this class are unused but kept for reference
# Only create_per_task_accuracy_graph, create_cluster_visualization,
# create_overall_accuracy_graph, and generate_simple_report are actively used


class TAGemVisualizer:
    def __init__(self):
        self.task_accuracies = []
        self.per_task_accuracies = []
        self.task_losses = []
        self.memory_sizes = []
        self.epoch_losses = []
        self.batch_losses = []
        self.training_times = []
        self.memory_efficiency = []
        self.learning_rates = []

    def update_metrics(
        self,
        task_id,
        overall_accuracy,
        individual_accuracies,
        epoch_losses,
        memory_size,
        training_time=None,
        learning_rate=None,
    ):
        """Update metrics for a completed task"""
        self.task_accuracies.append(overall_accuracy)
        self.per_task_accuracies.append(individual_accuracies.copy())
        self.memory_sizes.append(memory_size)
        self.epoch_losses.append(epoch_losses.copy())

        if learning_rate is not None:
            self.learning_rates.append(learning_rate)

        if training_time:
            self.training_times.append(training_time)

        # Calculate memory efficiency (accuracy per memory unit)
        if memory_size > 0:
            efficiency = overall_accuracy / memory_size
            self.memory_efficiency.append(efficiency)
        else:
            self.memory_efficiency.append(0)

    def add_batch_loss(self, task_id, epoch, batch_idx, loss):
        """Add individual batch loss for detailed analysis"""
        self.batch_losses.append(
            {"task": task_id, "epoch": epoch, "batch": batch_idx, "loss": loss}
        )

    def save_metrics(self, filepath, params=None):
        """Save all metrics and params to file for later analysis"""
        metrics = {
            "task_accuracies": self.task_accuracies,
            "per_task_accuracies": self.per_task_accuracies,
            "task_losses": self.task_losses,
            "memory_sizes": self.memory_sizes,
            "epoch_losses": self.epoch_losses,
            "batch_losses": self.batch_losses,
            "training_times": self.training_times,
            "memory_efficiency": self.memory_efficiency,
            "learning_rate": self.learning_rates,
            "timestamp": datetime.now().isoformat(),
        }

        if params:
            metrics["params"] = params

        with open(filepath, "wb") as f:
            pickle.dump(metrics, f)
        print(f"Metrics saved to {filepath}")

    def load_metrics(self, filepath):
        """Load metrics from file"""
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                metrics = pickle.load(f)

            self.task_accuracies = metrics.get("task_accuracies", [])
            self.per_task_accuracies = metrics.get("per_task_accuracies", [])
            self.task_losses = metrics.get("task_losses", [])
            self.memory_sizes = metrics.get("memory_sizes", [])
            self.epoch_losses = metrics.get("epoch_losses", [])
            self.batch_losses = metrics.get("batch_losses", [])
            self.training_times = metrics.get("training_times", [])
            self.memory_efficiency = metrics.get("memory_efficiency", [])
            self.learning_rates = metrics.get("learning_rate", [])

            print(f"Metrics loaded from {filepath}")
            return True
        return False

    def create_per_task_accuracy_graph(self):
        """Create line graph showing accuracy per task over time, sampled per epoch"""
        if not self.per_task_accuracies:
            print("No per-task accuracy data available")
            return None

        plot_data = []
        for task_id, task_accuracies in enumerate(self.per_task_accuracies):
            for prev_task_id, accuracy in enumerate(task_accuracies):
                plot_data.append(
                    {
                        "Task_Learned": task_id,
                        "Task_Evaluated": prev_task_id,
                        "Accuracy": accuracy,
                    }
                )

        df = pd.DataFrame(plot_data)
        fig = px.line(
            df,
            x="Task_Learned",
            y="Accuracy",
            color="Task_Evaluated",
            title="Per-Task Accuracy Over Time",
            labels={
                "Task_Learned": "Training Progress (Tasks Completed)",
                "Task_Evaluated": "Task Being Evaluated",
            },
            range_y=[0.0, 1.0],
        )
        return fig

    def create_cluster_visualization(self, clustering_memory, clusters_to_show):
        """Create 3D visualization of cluster storage using multi-pool ClusteringMechanism.visualize()"""
        try:
            pools = clustering_memory.get_clustering_mechanism()
            if not pools:
                print("No clustering pools available for visualization")
                return

            print(f"Visualizing {clusters_to_show}/{len(pools)} clustering pools:")
            i = 0
            for label, pool in pools.items():
                if i >= clusters_to_show:
                    break
                pool.visualize()
                i += 1
        except Exception as e:
            print(f"Error creating cluster visualization: {e}")

    def create_overall_accuracy_graph(self):
        """Create line graph showing overall accuracy over time, sampled per epoch"""
        if not self.task_accuracies:
            print("No overall accuracy data available")
            return None

        fig = px.line(
            x=range(len(self.task_accuracies)),
            y=self.task_accuracies,
            title="Overall Accuracy Over Time",
            labels={"x": "Tasks Completed", "y": "Overall Accuracy"},
            range_y=[0.0, 1.0],
        )
        return fig

    def generate_simple_report(
        self, clustering_memory, clusters_to_show=1, show_images=True, save_path=None
    ):
        """Generate simplified analysis with just 3 graphs"""
        if not any([self.task_accuracies, self.per_task_accuracies]):
            print("No data available for report generation")
            return

        print("Generating simplified visualizations...")

        # Graph 1: Per-task accuracy over time
        try:
            per_task_fig = self.create_per_task_accuracy_graph()
            if per_task_fig:
                if show_images:
                    per_task_fig.show()
                if save_path is not None:
                    per_task_fig.write_html(f"{save_path}_per_task_accuracy.html")
                    print(
                        f"Per-task accuracy graph saved to {save_path}_per_task_accuracy.html"
                    )
        except Exception as e:
            print(f"Error creating per-task accuracy graph: {e}")

        # Graph 2: Cluster storage visualization
        try:
            print("Displaying cluster storage visualization...")
            if show_images:
                self.create_cluster_visualization(clustering_memory, clusters_to_show)
        except Exception as e:
            print(f"Error creating cluster visualization: {e}")

        # Graph 3: Overall accuracy over time
        try:
            overall_fig = self.create_overall_accuracy_graph()
            if overall_fig:
                if show_images:
                    overall_fig.show()
                if save_path is not None:
                    overall_fig.write_html(f"{save_path}_overall_accuracy.html")
                    print(
                        f"Overall accuracy graph saved to {save_path}_overall_accuracy.html"
                    )
        except Exception as e:
            print(f"Error creating overall accuracy graph: {e}")


class _Time:
    def __init__(self, start):
        self.start = start
        self.finish = None

    def __str__(self):
        return str(self.duration()) if self.finish is not None else ""

    def end(self, end):
        assert self.finish is None, "End time already set"
        self.finish = end

    def duration(self):
        return self.finish - self.start if self.finish is not None else None


class Timer:
    def __init__(self):
        self.times = {}

    def __str__(self):
        result = []
        for key in self.times:
            if isinstance(self.times[key], list):
                durations = [t.duration() for t in self.times[key]]
                count = len(durations)
                total = sum(durations)
                avg = total / count
                low = min(durations)
                high = max(durations)
                result.append(
                    f"{key}: total={total:.6f}, count={count}, avg={avg:.6f}, low={low:.6f}, high={high:.6f}"
                )
            else:
                result.append(f"{key}: {self.times[key]}")
        return "\n".join(result)

    def start(self, key):
        if key in self.times:
            if isinstance(self.times[key], list):
                # Key already has multiple measurements
                self.times[key].append(_Time(time.time()))
            else:
                # Key has single measurement, convert to list
                assert (
                    self.times[key].finish is not None
                ), f"Previous timer for key '{key}' not ended"
                self.times[key] = [self.times[key], _Time(time.time())]
        else:
            self.times[key] = _Time(time.time())

    def end(self, key):
        if isinstance(self.times[key], list):
            self.times[key][-1].end(time.time())
        else:
            self.times[key].end(time.time())


# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    t = Timer()
    t.start("1")
    time.sleep(2.5)
    t.start("2")
    time.sleep(0.5)
    t.end("2")
    t.end("1")
    print(t)

    # Create sample data for testing
    visualizer = TAGemVisualizer()

    # Simulate some training data
    np.random.seed(42)  # For reproducible test data
    for task_id in range(5):
        overall_acc = max(0.1, 0.8 - 0.05 * task_id + np.random.normal(0, 0.02))
        individual_accs = [
            max(0.1, 0.9 - 0.03 * max(0, task_id - i)) for i in range(task_id + 1)
        ]
        epoch_losses = [
            max(0.01, 1.0 * np.exp(-0.1 * epoch) + np.random.normal(0, 0.01))
            for epoch in range(20)
        ]
        memory_size = min(100, 20 * (task_id + 1))

        visualizer.update_metrics(
            task_id, overall_acc, individual_accs, epoch_losses, memory_size
        )

        # Add some batch losses
        for epoch in range(5):
            for batch in range(10):
                loss = max(0.01, epoch_losses[epoch] + np.random.normal(0, 0.05))
                visualizer.add_batch_loss(task_id, epoch, batch, loss)

    # Generate report
    visualizer.generate_report("test_analysis")

    # Save and load test
    visualizer.save_metrics("test_metrics.pkl")
    new_visualizer = TAGemVisualizer()
    new_visualizer.load_metrics("test_metrics.pkl")
