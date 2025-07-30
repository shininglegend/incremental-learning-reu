import math
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
import os
from datetime import datetime
import time

# MLflow integration
try:
    import mlflow
    import mlflow.pytorch

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available. Install with: conda install mlflow")

# NOTE: Many functions in this class are unused but kept for reference
# Only create_per_task_accuracy_graph, create_cluster_visualization,
# create_overall_accuracy_graph, and generate_simple_report are actively used


class TAGemVisualizer:
    def __init__(
        self,
        use_mlflow=True,
        experiment_name="Unnamed",
        total_samples=0,
        batch_size=10,
        sampling_rate=1,
    ):
        self.epoch_data = []  # List of dicts with epoch-level metrics
        self.task_boundaries = []  # Track where each task ends
        self.batch_losses = []
        self.training_times = []
        self.current_epoch = 0
        self._oldest_task_ids_tracking = []  # Track oldest task IDs matrix over time

        # MLflow configuration
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.experiment_name = experiment_name
        self.current_run = None
        self.global_step = 0

        if self.use_mlflow:
            self._setup_mlflow()

        # Settings for generating the line graph of when a task is "forgotten"
        frames_wanted = 500  # Number of x-axis frames desired
        self.tracking_interval = math.ceil(
            ((total_samples / batch_size) * sampling_rate) / frames_wanted
        )
        self.tracking_frames_seen = 0

    @property
    def task_accuracies(self):
        """Backward compatibility: return overall accuracies"""
        return [ep["overall_accuracy"] for ep in self.epoch_data]

    @property
    def per_task_accuracies(self):
        """Backward compatibility: return individual accuracies"""
        return [ep["individual_accuracies"] for ep in self.epoch_data]

    @property
    def memory_sizes(self):
        """Backward compatibility: return memory sizes"""
        return [ep["memory_size"] for ep in self.epoch_data]

    @property
    def epoch_losses(self):
        """Backward compatibility: return epoch losses"""
        return [[ep["epoch_loss"]] for ep in self.epoch_data if ep.get("epoch_loss", None) is not None]

    @property
    def learning_rates(self):
        """Backward compatibility: return learning rates"""
        return [ep["learning_rate"] for ep in self.epoch_data if ep["learning_rate"]]

    @property
    def memory_efficiency(self):
        """Backward compatibility: return memory efficiency"""
        return [
            ep["overall_accuracy"] / ep["memory_size"] if ep["memory_size"] > 0 else 0.0
            for ep in self.epoch_data
        ]

    @property
    def oldest_task_ids_tracking(self):
        """Return the oldest task IDs tracking data"""
        return self._oldest_task_ids_tracking

    @oldest_task_ids_tracking.setter
    def oldest_task_ids_tracking(self, value):
        """Set the oldest task IDs tracking data"""
        self._oldest_task_ids_tracking = value

    def _setup_mlflow(self):
        """Initialize MLflow experiment"""
        if not self.use_mlflow:
            return

        try:
            mlflow.set_experiment(self.experiment_name)
            print(f"MLflow experiment set to: {self.experiment_name}")
        except Exception as e:
            print(f"Failed to setup MLflow: {e}")
            self.use_mlflow = False

    def start_run(self, run_name=None, params=None):
        """Start a new MLflow run"""
        if not self.use_mlflow:
            return

        try:
            self.current_run = mlflow.start_run(run_name=run_name)

            # Log parameters if provided
            if params:
                mlflow.log_params(params)

            print(f"Started MLflow run: {self.current_run.info.run_id}")
        except Exception as e:
            print(f"Failed to start MLflow run: {e}")

    def end_run(self):
        """End the current MLflow run"""
        if not self.use_mlflow or not self.current_run:
            return

        try:
            mlflow.end_run()
            print(f"Ended MLflow run: {self.current_run.info.run_id}")
            self.current_run = None
        except Exception as e:
            print(f"Failed to end MLflow run: {e}")

    def log_model(self, model, task_id, epoch=None):
        """Log model to MLflow"""
        if not self.use_mlflow or not self.current_run:
            return

        try:
            artifact_path = f"model_task_{task_id}"
            if epoch is not None:
                artifact_path += f"_epoch_{epoch}"

            mlflow.pytorch.log_model(model, artifact_path)
            print(f"Model logged to MLflow: {artifact_path}")
        except Exception as e:
            print(f"Failed to log model to MLflow: {e}")

    def update_metrics(
        self,
        task_id,
        overall_accuracy,
        individual_accuracies,
        epoch_loss,
        memory_size,
        training_time=None,
        learning_rate=None,
    ):
        """Update metrics for a completed epoch"""
        epoch_data = {
            "epoch": self.current_epoch,
            "task_id": task_id,
            "overall_accuracy": overall_accuracy,
            "individual_accuracies": individual_accuracies.copy(),
            "memory_size": memory_size,
            "learning_rate": learning_rate,
            "training_time": training_time,
        }
        if epoch_loss is not None:
            epoch_data["epoch_loss"] = epoch_loss

        self.epoch_data.append(epoch_data)

        # Log to MLflow
        if self.use_mlflow and self.current_run:
            try:
                metrics = {
                    "overall_accuracy": overall_accuracy,
                    "memory_size": memory_size,
                    "current_task": task_id,
                }

                if learning_rate is not None:
                    metrics["learning_rate"] = learning_rate
                if training_time is not None:
                    metrics["training_time"] = training_time
                if epoch_loss is not None:
                    epoch_data["epoch_loss"] = epoch_loss

                # Log individual task accuracies
                for i, acc in enumerate(individual_accuracies):
                    metrics[f"task_{i}_accuracy"] = acc

                mlflow.log_metrics(metrics, step=self.global_step)
                self.global_step += 1

            except Exception as e:
                print(f"Failed to log metrics to MLflow: {e}")

        self.current_epoch += 1

        # Track task boundaries for visualization
        if training_time is not None:  # End of task
            self.task_boundaries.append(self.current_epoch - 1)
            if training_time:
                self.training_times.append(training_time)

    def add_batch_loss(self, task_id, epoch, batch_idx, loss):
        """Add individual batch loss for detailed analysis"""
        self.batch_losses.append(
            {"task": task_id, "epoch": epoch, "batch": batch_idx, "loss": loss}
        )

    def track_oldest_task_ids(self, clustering_memory, task_id):
        """Track the oldest task IDs matrix from clustering memory at reduced rate"""
        self.tracking_frames_seen += 1
        if self.tracking_frames_seen % self.tracking_interval == 0:
            matrix = clustering_memory.get_oldest_task_ids_matrix()
            self._oldest_task_ids_tracking.append((matrix, task_id))

    def save_metrics(self, filepath, params=None):
        """Save all metrics and params to file for later analysis"""
        metrics = {
            "epoch_data": self.epoch_data,
            "task_boundaries": self.task_boundaries,
            "batch_losses": self.batch_losses,
            "training_times": self.training_times,
            "oldest_task_ids_tracking": self._oldest_task_ids_tracking,
            "timestamp": datetime.now().isoformat(),
        }
        print(
            f"Saved {len(self._oldest_task_ids_tracking)} frames, saw {self.tracking_frames_seen}."
        )

        # Add legacy format for backward compatibility
        if self.epoch_data:
            metrics["task_accuracies"] = self.task_accuracies
            metrics["per_task_accuracies"] = self.per_task_accuracies
            metrics["memory_sizes"] = self.memory_sizes
            metrics["epoch_losses"] = self.epoch_losses
            metrics["learning_rate"] = self.learning_rates
            metrics["memory_efficiency"] = self.memory_efficiency

        if params:
            params["tracking_interval"] = self.tracking_interval
            metrics["params"] = params

        with open(filepath, "wb") as f:
            pickle.dump(metrics, f)
        print(f"Metrics saved to {filepath}")

        # Log as MLflow artifact
        if self.use_mlflow and self.current_run:
            try:
                mlflow.log_artifact(filepath, "metrics")
                print(f"Metrics also logged to MLflow as artifact")
            except Exception as e:
                print(f"Failed to log metrics to MLflow: {e}")

    def load_metrics(self, filepath):
        """Load metrics from file"""
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                metrics = pickle.load(f)

            self.epoch_data = metrics.get("epoch_data", [])
            self.task_boundaries = metrics.get("task_boundaries", [])
            self.batch_losses = metrics.get("batch_losses", [])
            self.training_times = metrics.get("training_times", [])
            self._oldest_task_ids_tracking = metrics.get("oldest_task_ids_tracking", [])
            self.current_epoch = len(self.epoch_data)

            print(f"Metrics loaded from {filepath}")
            return True
        return False

    def create_per_task_accuracy_graph(self):
        """Create line graph showing accuracy per task over time, sampled per epoch"""
        if not self.epoch_data:
            print("No epoch data available")
            return None

        plot_data = []
        for epoch_idx, epoch_info in enumerate(self.epoch_data):
            for task_id, accuracy in enumerate(epoch_info["individual_accuracies"]):
                plot_data.append(
                    {
                        "Epoch": epoch_idx,
                        "Task_Evaluated": task_id,
                        "Accuracy": accuracy,
                        "Current_Task": epoch_info["task_id"],
                    }
                )

        df = pd.DataFrame(plot_data)
        fig = px.line(
            df,
            x="Epoch",
            y="Accuracy",
            color="Task_Evaluated",
            title="Per-Task Accuracy Over Training Epochs",
            labels={
                "Epoch": "Training Epoch",
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
        if not self.epoch_data:
            print("No epoch data available")
            return None

        epochs = [ep["epoch"] for ep in self.epoch_data]
        accuracies = [ep["overall_accuracy"] for ep in self.epoch_data]

        fig = px.line(
            x=epochs,
            y=accuracies,
            title="Overall Accuracy Over Training Epochs",
            labels={"x": "Training Epoch", "y": "Overall Accuracy"},
            range_y=[0.0, 1.0],
        )
        return fig

    def generate_simple_report(
        self, clustering_memory, clusters_to_show=1, show_images=True, save_path=None
    ):
        """Generate simplified analysis with just 3 graphs"""
        if not self.epoch_data:
            print("No epoch data available for report generation")
            return

        print("Generating simplified visualizations...")

        # Graph 1: Per-task accuracy over time
        try:
            per_task_fig = self.create_per_task_accuracy_graph()
            if per_task_fig:
                if show_images:
                    per_task_fig.show()
                if save_path is not None:
                    html_path = f"{save_path}_per_task_accuracy.html"
                    per_task_fig.write_html(html_path)
                    print(f"Per-task accuracy graph saved to {html_path}")

                    # Log to MLflow
                    if self.use_mlflow and self.current_run:
                        try:
                            mlflow.log_artifact(html_path, "visualizations")
                        except Exception as e:
                            print(f"Failed to log visualization to MLflow: {e}")

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
                    html_path = f"{save_path}_overall_accuracy.html"
                    overall_fig.write_html(html_path)
                    print(f"Overall accuracy graph saved to {html_path}")

                    # Log to MLflow
                    if self.use_mlflow and self.current_run:
                        try:
                            mlflow.log_artifact(html_path, "visualizations")
                        except Exception as e:
                            print(f"Failed to log visualization to MLflow: {e}")

        except Exception as e:
            print(f"Error creating overall accuracy graph: {e}")

        # Log final summary metrics to MLflow
        if self.use_mlflow and self.current_run and self.epoch_data:
            try:
                final_accuracy = self.epoch_data[-1]["overall_accuracy"]
                final_memory_size = self.epoch_data[-1]["memory_size"]
                avg_accuracy = sum(
                    ep["overall_accuracy"] for ep in self.epoch_data
                ) / len(self.epoch_data)

                summary_metrics = {
                    "final_overall_accuracy": final_accuracy,
                    "final_memory_size": final_memory_size,
                    "average_accuracy_all_epochs": avg_accuracy,
                    "total_epochs": len(self.epoch_data),
                    "total_tasks": max(ep["task_id"] for ep in self.epoch_data) + 1,
                }

                mlflow.log_metrics(summary_metrics)
                print("Summary metrics logged to MLflow")

            except Exception as e:
                print(f"Failed to log summary metrics to MLflow: {e}")


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
