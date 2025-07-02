import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime

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

    def update_metrics(self, task_id, overall_accuracy, individual_accuracies,
                      epoch_losses, memory_size, training_time=None):
        """Update metrics for a completed task"""
        self.task_accuracies.append(overall_accuracy)
        self.per_task_accuracies.append(individual_accuracies.copy())
        self.memory_sizes.append(memory_size)
        self.epoch_losses.append(epoch_losses.copy())

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
        self.batch_losses.append({
            'task': task_id,
            'epoch': epoch,
            'batch': batch_idx,
            'loss': loss
        })

    def save_metrics(self, filepath, params=None):
        """Save all metrics and params to file for later analysis"""
        metrics = {
            'task_accuracies': self.task_accuracies,
            'per_task_accuracies': self.per_task_accuracies,
            'task_losses': self.task_losses,
            'memory_sizes': self.memory_sizes,
            'epoch_losses': self.epoch_losses,
            'batch_losses': self.batch_losses,
            'training_times': self.training_times,
            'memory_efficiency': self.memory_efficiency,
            'timestamp': datetime.now().isoformat()
        }

        if params:
            metrics['params'] = params

        with open(filepath, 'wb') as f:
            pickle.dump(metrics, f)
        print(f"Metrics saved to {filepath}")

    def load_metrics(self, filepath):
        """Load metrics from file"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                metrics = pickle.load(f)

            self.task_accuracies = metrics.get('task_accuracies', [])
            self.per_task_accuracies = metrics.get('per_task_accuracies', [])
            self.task_losses = metrics.get('task_losses', [])
            self.memory_sizes = metrics.get('memory_sizes', [])
            self.epoch_losses = metrics.get('epoch_losses', [])
            self.batch_losses = metrics.get('batch_losses', [])
            self.training_times = metrics.get('training_times', [])
            self.memory_efficiency = metrics.get('memory_efficiency', [])

            print(f"Metrics loaded from {filepath}")
            return True
        return False

    def create_main_dashboard(self):  # UNUSED - kept for reference
        """Create the main analysis dashboard"""
        # Validate data before creating visualizations
        if not any([self.task_accuracies, self.memory_sizes, self.epoch_losses]):
            print("Warning: No data available for main dashboard")
            return None

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Overall Performance Trajectory', 'Task-Specific Accuracy Evolution',
                           'Memory Utilization & Efficiency', 'Loss Convergence Analysis',
                           'Catastrophic Forgetting Heatmap', 'Training Dynamics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # 1. Overall performance trajectory with confidence intervals
        if self.task_accuracies:
            x_vals = list(range(len(self.task_accuracies)))
            fig.add_trace(
                go.Scatter(x=x_vals, y=self.task_accuracies,
                          mode='lines+markers', name='Overall Accuracy',
                          line=dict(color='#2E86AB', width=3),
                          marker=dict(size=8, symbol='circle')),
                row=1, col=1
            )

            # Add trend line
            if len(self.task_accuracies) > 2:
                z = np.polyfit(x_vals, self.task_accuracies, 1)
                p = np.poly1d(z)
                fig.add_trace(
                    go.Scatter(x=x_vals, y=p(x_vals),
                              mode='lines', name='Trend',
                              line=dict(color='red', dash='dash')),
                    row=1, col=1
                )

        # 2. Task-specific accuracy evolution
        if self.per_task_accuracies and len(self.per_task_accuracies) > 0:
            colors = px.colors.qualitative.Set1
            max_tasks = max(len(accs) for accs in self.per_task_accuracies) if self.per_task_accuracies else 0
            for task_id in range(max_tasks):
                task_accs = [accs[task_id] if task_id < len(accs) else 0
                            for accs in self.per_task_accuracies]
                if any(acc > 0 for acc in task_accs):  # Only plot if there's meaningful data
                    fig.add_trace(
                        go.Scatter(x=list(range(len(task_accs))), y=task_accs,
                                  mode='lines+markers', name=f'Task {task_id}',
                                  line=dict(color=colors[task_id % len(colors)], width=2)),
                        row=1, col=2
                    )

        # 3. Memory utilization and efficiency
        if self.memory_sizes:
            fig.add_trace(
                go.Scatter(x=list(range(len(self.memory_sizes))), y=self.memory_sizes,
                          mode='lines+markers', name='Memory Size',
                          line=dict(color='#A23B72', width=2)),
                row=2, col=1
            )

        if self.memory_efficiency:
            fig.add_trace(
                go.Scatter(x=list(range(len(self.memory_efficiency))), y=self.memory_efficiency,
                          mode='lines+markers', name='Memory Efficiency',
                          line=dict(color='#F18F01', width=2)),
                row=2, col=1, secondary_y=True
            )

        # 4. Loss convergence analysis
        if self.epoch_losses:
            colors = px.colors.qualitative.Pastel1
            for task_id, losses in enumerate(self.epoch_losses):
                if losses and len(losses) > 0:
                    # Ensure losses are valid numbers
                    valid_losses = [l for l in losses if isinstance(l, (int, float)) and not np.isnan(l)]
                    if valid_losses:
                        fig.add_trace(
                            go.Scatter(x=list(range(len(valid_losses))), y=valid_losses,
                                      mode='lines', name=f'Task {task_id} Loss',
                                      line=dict(color=colors[task_id % len(colors)])),
                            row=2, col=2
                        )

        # 5. Catastrophic forgetting heatmap
        if len(self.per_task_accuracies) > 1:
            forgetting_matrix = self._compute_forgetting_matrix()
            fig.add_trace(
                go.Heatmap(z=forgetting_matrix,
                          colorscale='RdBu',
                          zmid=0,
                          showscale=True,
                          colorbar=dict(title="Forgetting", x=0.48)),
                row=3, col=1
            )

        # 6. Training dynamics (batch loss smoothed)
        if self.batch_losses and len(self.batch_losses) > 0:
            try:
                batch_df = pd.DataFrame(self.batch_losses)
                if not batch_df.empty and 'task' in batch_df.columns and 'loss' in batch_df.columns:
                    for task_id in batch_df['task'].unique():
                        task_data = batch_df[batch_df['task'] == task_id]
                        # Smooth with rolling average
                        if len(task_data) > 10:
                            # Remove invalid values before smoothing
                            task_data = task_data[task_data['loss'].notna()]
                            if len(task_data) > 5:
                                smoothed = task_data['loss'].rolling(window=min(10, len(task_data)), center=True).mean()
                                smoothed = smoothed.dropna()
                                if len(smoothed) > 0:
                                    fig.add_trace(
                                        go.Scatter(x=list(range(len(smoothed))), y=list(smoothed),
                                                  mode='lines', name=f'Task {task_id} Smoothed',
                                                  line=dict(width=2)),
                                        row=3, col=2
                                    )
            except Exception as e:
                print(f"Warning: Could not create batch loss visualization: {e}")

        # Update layout with professional styling
        fig.update_layout(
            height=1200, width=1600,
            title=dict(
                text="TA-A-GEM Comprehensive Training Analysis",
                x=0.5,
                font=dict(size=20, color='#2E4057')
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='white',
            paper_bgcolor='#FAFAFA'
        )

        # Update axes styling
        for i in range(1, 4):
            for j in range(1, 3):
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray',
                               row=i, col=j)
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray',
                               row=i, col=j)

        # Specific axis labels
        fig.update_xaxes(title_text="Training Task", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_xaxes(title_text="Training Task", row=1, col=2)
        fig.update_yaxes(title_text="Task Accuracy", row=1, col=2)
        fig.update_xaxes(title_text="Training Task", row=2, col=1)
        fig.update_yaxes(title_text="Memory Size", row=2, col=1)
        fig.update_yaxes(title_text="Efficiency (Acc/Memory)", row=2, col=1, secondary_y=True)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        fig.update_yaxes(title_text="Loss", row=2, col=2)
        fig.update_xaxes(title_text="Learned Task", row=3, col=1)
        fig.update_yaxes(title_text="Evaluation Task", row=3, col=1)
        fig.update_xaxes(title_text="Batch (Smoothed)", row=3, col=2)
        fig.update_yaxes(title_text="Loss", row=3, col=2)

        return fig

    def create_forgetting_analysis(self):  # UNUSED - kept for reference
        """Create detailed catastrophic forgetting analysis"""
        if len(self.per_task_accuracies) < 2:
            print("Warning: Need at least 2 tasks for forgetting analysis")
            return None

        forgetting_data = self._compute_detailed_forgetting()
        if not forgetting_data or all(len(task_data) == 0 for task_data in forgetting_data):
            print("Warning: No forgetting data available")
            return None

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Forgetting Timeline', 'Final vs Initial Performance',
                           'Forgetting Distribution', 'Retention Scores'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # 1. Forgetting timeline
        for task_id in range(len(forgetting_data)):
            task_forget = [fd['forgetting'] for fd in forgetting_data[task_id]]
            if task_forget:  # Only add if there's data
                fig.add_trace(
                    go.Scatter(x=list(range(len(task_forget))), y=task_forget,
                              mode='lines+markers', name=f'Task {task_id}'),
                    row=1, col=1
                )

        # 2. Final vs Initial performance scatter
        initial_accs = []
        final_accs = []
        task_ids = []

        for task_id in range(len(self.per_task_accuracies) - 1):
            if task_id < len(self.per_task_accuracies[task_id]):
                initial_acc = self.per_task_accuracies[task_id][task_id]
                final_acc = self.per_task_accuracies[-1][task_id]
                initial_accs.append(initial_acc)
                final_accs.append(final_acc)
                task_ids.append(task_id)

        if initial_accs:
            fig.add_trace(
                go.Scatter(x=initial_accs, y=final_accs,
                          mode='markers', name='Tasks',
                          marker=dict(size=10, opacity=0.7),
                          text=[f'Task {tid}' for tid in task_ids],
                          textposition='top center'),
                row=1, col=2
            )
            # Add diagonal line
            max_acc = max(max(initial_accs), max(final_accs))
            fig.add_trace(
                go.Scatter(x=[0, max_acc], y=[0, max_acc],
                          mode='lines', name='No Forgetting',
                          line=dict(dash='dash', color='red')),
                row=1, col=2
            )

        # 3. Forgetting distribution
        all_forgetting = []
        for task_data in forgetting_data:
            all_forgetting.extend([fd['forgetting'] for fd in task_data])

        if all_forgetting:
            fig.add_trace(
                go.Histogram(x=all_forgetting, nbinsx=20, name='Forgetting Distribution'),
                row=2, col=1
            )

        # 4. Retention scores (1 - forgetting)
        retention_scores = []
        for task_id in range(len(forgetting_data)):
            if forgetting_data[task_id]:
                final_forgetting = forgetting_data[task_id][-1]['forgetting']
                retention = max(0, 1 - final_forgetting)
                retention_scores.append(retention)

        if retention_scores:
            fig.add_trace(
                go.Bar(x=list(range(len(retention_scores))), y=retention_scores,
                      name='Retention Score',
                      marker_color=['green' if rs > 0.8 else 'orange' if rs > 0.6 else 'red'
                                  for rs in retention_scores]),
                row=2, col=2
            )

        fig.update_layout(
            height=800, width=1200,
            title_text="Catastrophic Forgetting Detailed Analysis",
            showlegend=True
        )

        return fig

    def create_memory_analysis(self):  # UNUSED - kept for reference
        """Create memory utilization analysis"""
        if not self.memory_sizes or len(self.memory_sizes) == 0:
            print("Warning: No memory data available for analysis")
            return None

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Memory Growth', 'Memory Efficiency',
                           'Accuracy vs Memory', 'Memory Utilization Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # 1. Memory growth
        fig.add_trace(
            go.Scatter(x=list(range(len(self.memory_sizes))), y=self.memory_sizes,
                      mode='lines+markers', name='Memory Size',
                      fill='tonexty', line=dict(color='blue')),
            row=1, col=1
        )

        # 2. Memory efficiency
        if self.memory_efficiency:
            fig.add_trace(
                go.Scatter(x=list(range(len(self.memory_efficiency))), y=self.memory_efficiency,
                          mode='lines+markers', name='Efficiency',
                          line=dict(color='green')),
                row=1, col=2
            )

        # 3. Accuracy vs Memory scatter
        if len(self.task_accuracies) == len(self.memory_sizes):
            fig.add_trace(
                go.Scatter(x=self.memory_sizes, y=self.task_accuracies,
                          mode='markers', name='Accuracy vs Memory',
                          marker=dict(size=8, opacity=0.7)),
                row=2, col=1
            )

        # 4. Memory utilization rate (change in memory per task)
        if len(self.memory_sizes) > 1:
            mem_changes = [self.memory_sizes[i] - self.memory_sizes[i-1]
                          for i in range(1, len(self.memory_sizes))]
            fig.add_trace(
                go.Bar(x=list(range(1, len(self.memory_sizes))), y=mem_changes,
                      name='Memory Change per Task'),
                row=2, col=2
            )

        fig.update_layout(
            height=800, width=1200,
            title_text="Memory Utilization Analysis"
        )

        return fig

    def _compute_forgetting_matrix(self):  # UNUSED - kept for reference
        """Compute forgetting matrix for heatmap"""
        n_tasks = len(self.per_task_accuracies)
        forgetting_matrix = np.zeros((n_tasks, n_tasks))

        for eval_task in range(n_tasks):
            for learn_task in range(eval_task, n_tasks):
                if eval_task < len(self.per_task_accuracies[eval_task]):
                    initial_acc = self.per_task_accuracies[eval_task][eval_task]
                    if learn_task < len(self.per_task_accuracies) and eval_task < len(self.per_task_accuracies[learn_task]):
                        current_acc = self.per_task_accuracies[learn_task][eval_task]
                        forgetting = initial_acc - current_acc
                        forgetting_matrix[eval_task, learn_task] = forgetting

        return forgetting_matrix

    def _compute_detailed_forgetting(self):  # UNUSED - kept for reference
        """Compute detailed forgetting data for analysis"""
        forgetting_data = []

        for task_id in range(len(self.per_task_accuracies) - 1):
            task_forgetting = []
            if task_id < len(self.per_task_accuracies[task_id]):
                initial_acc = self.per_task_accuracies[task_id][task_id]

                for eval_step in range(task_id + 1, len(self.per_task_accuracies)):
                    if task_id < len(self.per_task_accuracies[eval_step]):
                        current_acc = self.per_task_accuracies[eval_step][task_id]
                        forgetting = initial_acc - current_acc
                        task_forgetting.append({
                            'step': eval_step,
                            'accuracy': current_acc,
                            'forgetting': forgetting
                        })

            forgetting_data.append(task_forgetting)

        return forgetting_data

    def create_per_task_accuracy_graph(self):
        """Create line graph showing accuracy per task over time, sampled per epoch"""
        if not self.per_task_accuracies:
            print("No per-task accuracy data available")
            return None

        plot_data = []
        for task_id, task_accuracies in enumerate(self.per_task_accuracies):
            for prev_task_id, accuracy in enumerate(task_accuracies):
                plot_data.append({
                    'Task_Learned': task_id,
                    'Task_Evaluated': prev_task_id,
                    'Accuracy': accuracy
                })

        df = pd.DataFrame(plot_data)
        fig = px.line(df, x='Task_Learned', y='Accuracy', color='Task_Evaluated',
                      title='Per-Task Accuracy Over Time',
                      labels={'Task_Learned': 'Training Progress (Tasks Completed)',
                             'Task_Evaluated': 'Task Being Evaluated'},
                      range_y=[0.0, 1.0])
        return fig

    def create_cluster_visualization(self, clustering_memory):
        """Create 3D visualization of cluster storage using multi-pool ClusteringMechanism.visualize()"""
        try:
            pools = clustering_memory.get_clustering_mechanism()
            if not pools:
                print("No clustering pools available for visualization")
                return

            print(f"Visualizing {len(pools)} clustering pools:")
            for label, pool in pools.items():
                print(f"Pool for label {label}:")
                pool.visualize()
        except Exception as e:
            print(f"Error creating cluster visualization: {e}")

    def create_overall_accuracy_graph(self):
        """Create line graph showing overall accuracy over time, sampled per epoch"""
        if not self.task_accuracies:
            print("No overall accuracy data available")
            return None

        fig = px.line(x=range(len(self.task_accuracies)), y=self.task_accuracies,
                     title='Overall Accuracy Over Time',
                     labels={'x': 'Tasks Completed', 'y': 'Overall Accuracy'},
                     range_y=[0.0, 1.0])
        return fig

    def generate_simple_report(self, clustering_memory, save_path=None):
        """Generate simplified analysis with just 3 graphs"""
        if not any([self.task_accuracies, self.per_task_accuracies]):
            print("No data available for report generation")
            return

        print("Generating simplified visualizations...")

        # Graph 1: Per-task accuracy over time
        try:
            per_task_fig = self.create_per_task_accuracy_graph()
            if per_task_fig:
                per_task_fig.show()
                if save_path:
                    per_task_fig.write_html(f"{save_path}_per_task_accuracy.html")
                    print(f"Per-task accuracy graph saved to {save_path}_per_task_accuracy.html")
        except Exception as e:
            print(f"Error creating per-task accuracy graph: {e}")

        # Graph 2: Cluster storage visualization
        try:
            print("Displaying cluster storage visualization...")
            self.create_cluster_visualization(clustering_memory)
        except Exception as e:
            print(f"Error creating cluster visualization: {e}")

        # Graph 3: Overall accuracy over time
        try:
            overall_fig = self.create_overall_accuracy_graph()
            if overall_fig:
                overall_fig.show()
                if save_path:
                    overall_fig.write_html(f"{save_path}_overall_accuracy.html")
                    print(f"Overall accuracy graph saved to {save_path}_overall_accuracy.html")
        except Exception as e:
            print(f"Error creating overall accuracy graph: {e}")

    def generate_report(self, save_path=None):  # UNUSED - kept for reference
        """Generate comprehensive analysis report"""
        if not any([self.task_accuracies, self.memory_sizes, self.epoch_losses]):
            print("No data available for report generation")
            return

        print("Generating visualizations...")

        # Create main dashboard
        try:
            main_fig = self.create_main_dashboard()
            if main_fig:
                main_fig.show()
                if save_path:
                    main_fig.write_html(f"{save_path}_main_dashboard.html")
                    print(f"Main dashboard saved to {save_path}_main_dashboard.html")
        except Exception as e:
            print(f"Error creating main dashboard: {e}")

        # Create forgetting analysis
        try:
            forget_fig = self.create_forgetting_analysis()
            if forget_fig:
                forget_fig.show()
                if save_path:
                    forget_fig.write_html(f"{save_path}_forgetting_analysis.html")
                    print(f"Forgetting analysis saved to {save_path}_forgetting_analysis.html")
        except Exception as e:
            print(f"Error creating forgetting analysis: {e}")

        # Create memory analysis
        try:
            memory_fig = self.create_memory_analysis()
            if memory_fig:
                memory_fig.show()
                if save_path:
                    memory_fig.write_html(f"{save_path}_memory_analysis.html")
                    print(f"Memory analysis saved to {save_path}_memory_analysis.html")
        except Exception as e:
            print(f"Error creating memory analysis: {e}")

        # Print summary statistics
        try:
            self._print_summary_stats()
        except Exception as e:
            print(f"Error printing summary stats: {e}")

    def _print_summary_stats(self):  # UNUSED - kept for reference
        """Print summary statistics"""
        print("\n" + "="*50)
        print("TA-A-GEM TRAINING SUMMARY")
        print("="*50)

        if self.task_accuracies:
            print(f"Final Overall Accuracy: {self.task_accuracies[-1]:.4f}")
            print(f"Best Overall Accuracy: {max(self.task_accuracies):.4f}")
            print(f"Accuracy Trend: {np.polyfit(range(len(self.task_accuracies)), self.task_accuracies, 1)[0]:.6f}")

        if self.memory_sizes:
            print(f"Final Memory Size: {self.memory_sizes[-1]} samples")
            print(f"Average Memory Growth: {(self.memory_sizes[-1] - self.memory_sizes[0]) / len(self.memory_sizes):.2f} samples/task")

        if self.memory_efficiency:
            print(f"Final Memory Efficiency: {self.memory_efficiency[-1]:.6f} accuracy/sample")

        if len(self.per_task_accuracies) > 1:
            # Calculate average forgetting
            total_forgetting = 0
            count = 0
            for task_id in range(len(self.per_task_accuracies) - 1):
                if task_id < len(self.per_task_accuracies[task_id]) and task_id < len(self.per_task_accuracies[-1]):
                    initial = self.per_task_accuracies[task_id][task_id]
                    final = self.per_task_accuracies[-1][task_id]
                    total_forgetting += (initial - final)
                    count += 1

            if count > 0:
                avg_forgetting = total_forgetting / count
                print(f"Average Catastrophic Forgetting: {avg_forgetting:.4f}")

        print("="*50)

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    visualizer = TAGemVisualizer()

    # Simulate some training data
    np.random.seed(42)  # For reproducible test data
    for task_id in range(5):
        overall_acc = max(0.1, 0.8 - 0.05 * task_id + np.random.normal(0, 0.02))
        individual_accs = [max(0.1, 0.9 - 0.03 * max(0, task_id - i)) for i in range(task_id + 1)]
        epoch_losses = [max(0.01, 1.0 * np.exp(-0.1 * epoch) + np.random.normal(0, 0.01)) for epoch in range(20)]
        memory_size = min(100, 20 * (task_id + 1))

        visualizer.update_metrics(task_id, overall_acc, individual_accs, epoch_losses, memory_size)

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
