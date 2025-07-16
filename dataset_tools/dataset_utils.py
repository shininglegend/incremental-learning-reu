import os
import kagglehub
from pathlib import Path

try:
    from dotenv import load_dotenv, set_key
except ImportError:
    load_dotenv = None
    set_key = None


def get_dataset_path(dataset_name, kaggle_dataset_id, path_override=None):
    """
    Get dataset path with env fallback and auto-download.

    Args:
        dataset_name: Name for env variable (e.g., 'MNIST', 'MNIST_FASHION')
        kaggle_dataset_id: Kaggle dataset identifier for download
        path: Path to the Kaggle dataset, will ignore the .env

    Returns:
        str: Path to dataset directory
    """
    if path_override is not None:
        path_override = os.path.join(path_override, dataset_name)
        assert Path(path_override).exists(), f"Dataset not found at `{path_override}`. Aborting"
        return path_override
    env_var = f"DATASET_LOCATION_{dataset_name}"
    env_file = Path(".env")

    # Load .env file if it exists
    if env_file.exists() and load_dotenv:
        load_dotenv(env_file)

    # Check if env variable is set and path exists
    dataset_path = os.getenv(env_var)
    if dataset_path and Path(dataset_path).exists():
        return dataset_path

    # Download dataset using kagglehub
    dataset_path = kagglehub.dataset_download(kaggle_dataset_id)

    # Update .env file
    if set_key:
        set_key(env_file, env_var, dataset_path)
    else:
        # Fallback if python-dotenv not available
        with open(env_file, "a") as f:
            f.write(f"{env_var}={dataset_path}\n")

    return dataset_path
