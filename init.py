import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from agem import agem
from agem.learning_rate import TALearningRateScheduler
from em_tools import clustering_pools
from dataset_tools.load_dataset import load_dataset
from visualization_analysis.visualization_analysis import TAGemVisualizer, Timer
from utils import task_introduction


def load_config(config_path="config/default.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TA-A-GEM Incremental Learning")
    parser.add_argument(
        "--config",
        type=str,
        default="utils/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        choices=["permutation", "rotation", "class_split"],
        help="Type of task transformation",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name for the experiment (used for MLflow experiment naming)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["mnist", "fashion_mnist", "cifar10"],
        help="Which dataset to use",
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        default=None,
        help="Run in quick test mode with fewer tasks and data",
    )
    parser.add_argument(
        "--no_verbose",
        action="store_true",
        default=False,
        help="Turn off printing dynamic progress reports for sbatch",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_results",
        help="Path to store the output in, should be a folder",
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset")
    return parser.parse_args()


def create_model(input_dim, hidden_dim, num_classes):
    """Create and return a simple MLP model."""

    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_classes):
            super(SimpleMLP, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            # Ensure consistent flattening regardless of input shape
            if len(x.shape) > 2:
                x = x.view(x.size(0), -1)  # Flatten to (batch_size, features)
            elif len(x.shape) == 1:
                x = x.unsqueeze(0)  # Add batch dimension if missing
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            return x

    return SimpleMLP(input_dim, hidden_dim, num_classes)


def initialize_system():
    """Initialize the complete TA-A-GEM system with configuration."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set some default values that aren't in the config file
    config["verbose"] = True
    config["data_dir"] = None

    # Override config with command line arguments
    if args.task_type is not None:
        config["task_type"] = args.task_type
    if args.lite is not None:
        config["lite"] = args.lite
    if args.dataset is not None:
        config["dataset_name"] = args.dataset
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.no_verbose is not None:
        config["verbose"] = not args.no_verbose  # Note: Inverted.
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir
    if args.experiment_name is not None:
        config["experiment_name"] = args.experiment_name

    # Sanity checks
    if config["sampling_rate"] > config["batch_size"]:
        raise Exception("Samping rate exceeds batch size.")
    if config["task_introduction"] not in [
        "sequential",
        "half and half",
        "random",
        "continuous",
    ]:
        raise Exception("Configuration task_introduction is invalid.")

    # Apply lite mode overrides
    if config["lite"]:
        config["num_tasks"] = 2
        config["batch_size"] = 50

    # Apply per-dataset overrides
    match config["dataset_name"]:
        case "mnist" | "fashion_mnist":
            config["input_dim"] = 784
            config["num_classes"] = 10
        case "cifar10":
            config["input_dim"] = 1024
            config["num_classes"] = 10
        case _:
            raise Exception("Dataset name not recognized.")

    # Determine configuration based on task type
    task_specific_config = config["task_specific"][config["task_type"]]
    config["num_pools"] = task_specific_config["num_pools"]
    config["clusters_per_pool"] = task_specific_config["clusters_per_pool"]
    config["num_tasks"] = task_specific_config["num_tasks"]

    # Create params dictionary for compatibility
    params = {
        "add_remove_randomly": config["add_remove_randomly"],
        "batch_size": config["batch_size"],
        "dataset_name": config["dataset_name"],
        "experiment_name": config["experiment_name"],
        "hidden_dim": config["hidden_dim"],
        "input_dim": config["input_dim"],
        "learning_rate": config["learning_rate"],
        "memory_size_p": config["memory_size_p"],
        "memory_size_q": config["clusters_per_pool"],
        "num_classes": config["num_classes"],
        "num_epochs": config["num_epochs"],
        "num_pools": config["num_pools"],
        "num_tasks": config["num_tasks"],
        "output_dir": config["output_dir"],
        "quick_test_mode": config["lite"],
        "random_em": config["random_em"],
        "sampling_rate": config["sampling_rate"],
        "task_type": config["task_type"],
        "task_introduction": config["task_introduction"],
        "use_lr_scheduler": config["use_learning_rate_scheduler"],
        "verbose": config["verbose"],
    }

    # Set up timer
    timer = Timer()
    timer.start("init")

    # Ensure output dir exists
    if not os.path.exists(config["output_dir"]) and os.path.isabs(config["output_dir"]):
        raise FileNotFoundError("path is an absolute path and doesn't exist.")
    os.makedirs(config["output_dir"], exist_ok=True)

    # Device configuration
    device = config["device"]
    if config["device"] == "cuda":
        # Intelligently overwrite the cuda setting if cuda isn't avaliable
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Initialize model, optimizer, and loss function
    model = create_model(
        config["input_dim"], config["hidden_dim"], config["num_classes"]
    ).to(device)
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Initialize learning rate scheduler
    lr_scheduler = (
        TALearningRateScheduler(lr_init=config["learning_rate"])
        if config["use_learning_rate_scheduler"]
        else None
    )

    # Initialize TA-A-GEM components
    clustering_memory = clustering_pools.ClusteringMemory(
        Q=config["clusters_per_pool"],
        P=config["memory_size_p"],
        input_type="samples",
        device=device,
        num_pools=config["num_pools"],
        add_remove_randomly=config["add_remove_randomly"],
    )

    # Get the current removal function and save it
    config["removal_function"] = clustering_memory.get_removal_function()
    params["removal_function"] = config["removal_function"]
    print(f"Removal function name: {config['removal_function']}")

    agem_handler = agem.AGEMHandler(
        model, criterion, optimizer, device=device, lr_scheduler=lr_scheduler
    )

    # Load dataset
    print("Loading dataset and preparing data loaders...")
    datasetLoader = load_dataset(config["dataset_name"], config["data_dir"])
    train_dataloaders, test_dataloaders = datasetLoader.prepare_domain_incremental_data(
        task_type=config["task_type"],
        num_tasks=config["num_tasks"],
        batch_size=config["batch_size"],
        quick_test=config["lite"],
        use_cuda=(device == "cuda"),
    )

    # Initialize visualizer with experiment name
    experiment_name = config.get("experiment_name", "Unnamed")
    visualizer = TAGemVisualizer(
        experiment_name=experiment_name,
        total_samples=sum(
            [(len(dl) * config["batch_size"]) for dl in train_dataloaders]
        ),
        batch_size=config["batch_size"],
        sampling_rate=config["sampling_rate"],
    )

    # Per-epoch main.py construction
    # epoch_list is a list of keys for the program to run in sequence
    # each item in the list corresponds to a key in train_dataloaders_dict
    # the value in train_dataloaders_dict is a dataloader corresponding to that key
    # iterate over epoch_list to get the epoch order for training.
    epoch_list, train_dataloaders_dict = task_introduction.get_epoch_order(
        train_dataloaders=train_dataloaders,
        num_tasks=config["num_tasks"],
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        num_transition_epochs=config["num_transition_epochs"],
        task_introduction=config["task_introduction"],
    )

    num_epochs_per_task = task_introduction.make_num_epochs_into_dict(
        num_epochs_per_task=config["num_epochs"], num_tasks=config["num_tasks"]
    )

    # Create run name with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{config['task_type']}_{timestamp}"

    # Start ml_flow run
    visualizer.start_run(
        run_name=run_name, params=config  # Log all configuration as parameters
    )

    timer.end("init")

    return (
        config,
        params,
        timer,
        device,
        model,
        optimizer,
        criterion,
        lr_scheduler,
        clustering_memory,
        agem_handler,
        train_dataloaders,  # dict
        test_dataloaders,  # list for legacy compatibility
        visualizer,
        epoch_list,
        train_dataloaders_dict,
        num_epochs_per_task,
    )
