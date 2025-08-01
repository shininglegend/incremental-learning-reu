# Written by Abigail Dodd
# Helper file for deciding task order
import math
import random
import torch
from torch.utils.data import DataLoader


def get_epoch_order(
    train_dataloaders,
    task_introduction,
    num_tasks,
    num_epochs,
    batch_size=None,
    num_transition_epochs=None,
):
    """
    Accepts configuration parameters directly

    Parameters:
    - task_introduction: str - type of task introduction ('sequential', 'half and half', 'random', 'continuous')
    - num_tasks: int - number of tasks
    - num_epochs: int, list, or dict - epochs per task
    - batch_size: int - batch size (required for 'continuous')
    - num_transition_epochs: int - transition epochs (required for 'continuous')

    Currently supports:
    'sequential' - epochs per task, introduced in sequence
    'half and half' - 1/2 of total epochs per task in sequence, then 1/2 total epochs again in task sequence
    'random' - num_epochs per task total, but the program will decide the order.
    'continuous' - continual learning with gradual transitions
    """

    if task_introduction == "continuous":
        if batch_size is None or num_transition_epochs is None:
            raise ValueError(
                "batch_size and num_transition_epochs required for continuous task introduction"
            )
        return continual_change(
            train_dataloaders=train_dataloaders,
            num_transition_epochs=num_transition_epochs,
            num_tasks=num_tasks,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

    task_epochs_dict = make_num_epochs_into_dict(
        num_epochs_per_task=num_epochs, num_tasks=num_tasks
    )
    train_dataloaders_dict = {
        index: value for index, value in enumerate(train_dataloaders)
    }

    if task_introduction == "sequential":
        return (
            _get_pure_sequential_order(task_epochs_dict=task_epochs_dict),
            train_dataloaders_dict,
        )
    elif task_introduction == "half and half":
        return (
            _get_half_and_half_epoch_order(task_epochs_dict=task_epochs_dict),
            train_dataloaders_dict,
        )
    elif task_introduction == "random":
        return (
            _get_random_epoch_order(task_epochs_dict=task_epochs_dict),
            train_dataloaders_dict,
        )
    else:
        raise Exception(
            f'Config value task_introduction is set to "{task_introduction}", which isn\'t currently supported.'
        )


def _get_random_epoch_order(task_epochs_dict):
    epoch_order = _get_pure_sequential_order(task_epochs_dict=task_epochs_dict)
    random.shuffle(epoch_order)
    return epoch_order


def _get_half_and_half_epoch_order(task_epochs_dict):
    """
    Gives an epoch order where the tasks are seen twice.
    """

    first_half = []
    second_half = []

    for task_id, epoch_num in task_epochs_dict.items():
        for i in range(math.ceil(epoch_num / 2)):
            first_half.append(task_id)
        for i in range(math.floor(epoch_num / 2)):
            second_half.append(task_id)

    epoch_order = first_half + second_half
    return epoch_order


def _get_pure_sequential_order(task_epochs_dict):
    """
    Takes in: dict of key:value where key is task_id and value is number of epochs for that id
    Returns a list that contains the id of the task, in the order that we want to do them.
    """


    epoch_order = []

    for task_id, num_epochs in task_epochs_dict.items():
        for _ in range(num_epochs):
            epoch_order.append(task_id)

    epoch_order.sort()  # make sure they're in task order.
    return epoch_order


def make_num_epochs_into_dict(num_epochs_per_task, num_tasks):
    """
    Checks the type of num_epochs_per_task and performs the necessary functions.
    """

    task_nums_and_epochs_dict = {}

    if isinstance(num_epochs_per_task, list):
        for task_id, num_epochs in enumerate(num_epochs_per_task):
            task_nums_and_epochs_dict[task_id] = num_epochs
    elif isinstance(num_epochs_per_task, int):
        # Assume every task is the same number of epochs.
        for i in range(num_tasks):
            task_nums_and_epochs_dict[i] = num_epochs_per_task

    elif isinstance(num_epochs_per_task, dict):
        task_nums_and_epochs_dict = num_epochs_per_task

    return task_nums_and_epochs_dict


def continual_change(
    train_dataloaders, num_transition_epochs, num_tasks, num_epochs, batch_size
):
    """
    Takes in num_tasks train_dataloaders and alters them for continual learning.
    See "section F.5. Continuous change experiments" in the Lamers appendix for more details.

    To better match lamers, we assume that continual change has an "overlap" of 10 epochs.
    So we alternate: 10 pure, 10 mixed, 10 pure, 10 mixed, ..... 10 pure, end with 10 final task to initial task mixed.
    """

    assert isinstance(
        num_transition_epochs, int
    ), "num_transition_epochs must be an integer for continuous change."
    assert (
        num_transition_epochs is not None
    ), "num_transition_epochs doesn't exist. Try again."
    assert num_transition_epochs >= 0, "Did not provide enough transition epochs"

    # Now we can assume num_transition_epochs is a positive integer
    mixed_dataloaders_dict = _get_continual_change_dataloaders(
        train_dataloaders=train_dataloaders,
        num_transition_epochs=num_transition_epochs,
        batch_size=batch_size,
    )

    mixed_epoch_list = []
    epochs_per_task = num_epochs

    for task_index in range(num_tasks):
        # Here, we assume num_epochs is equal for all tasks.
        for i in range(epochs_per_task - num_transition_epochs + 1):
            mixed_epoch_list.append(task_index)
        for i in range(num_transition_epochs - 1):
            mixed_epoch_list.append(task_index + (i + 1) / num_transition_epochs)

    return mixed_epoch_list, mixed_dataloaders_dict


def _get_continual_change_dataloaders(
    train_dataloaders, num_transition_epochs, batch_size
):

    mixed_dataloaders_dict = {}
    for task_id, task_dataloader in enumerate(train_dataloaders):
        mixed_dataloaders_dict[task_id] = task_dataloader

        next_task_id = task_id + 1
        if next_task_id >= len(train_dataloaders):
            next_task_id = -1

        current_dataloader = task_dataloader
        next_dataloader = train_dataloaders[next_task_id]

        for i in range(num_transition_epochs):
            mixed_dataloaders_dict[task_id + (i + 1) / num_transition_epochs] = (
                MixedTaskDataLoader(
                    dataloader1=current_dataloader,
                    task_id_1=task_id,
                    dataloader2=next_dataloader,
                    task_id_2=next_task_id,
                    proportion_task1=1 - (i + 1) / num_transition_epochs,
                    batch_size=batch_size,
                )

            )

    return mixed_dataloaders_dict


class MixedTaskDataLoader:
    """
    A custom dataloader that combines batches from two source dataloaders
    while maintaining a specified proportion of samples from each.
    The combined samples within each batch are shuffled.

    Args:
        dataloader1 (torch.utils.data.DataLoader): DataLoader for Task 1.
        task_id_1 (int): Task ID for task 1
        dataloader2 (torch.utils.data.DataLoader): DataLoader for Task 2.
        task_id_2 (int): Task ID for task 2
        proportion_task1 (float): The desired proportion of samples from Task 1
                                  in each combined batch (0.0 to 1.0).
                                  A proportion of 1.0 means all samples will be from Task 1.
        batch_size (int): The batch size for the combined batches.
                          This should match the batch sizes of dataloader1
                          and dataloader2, or be a multiple/divisor of them.
                          For simplicity, it's assumed to be the same as
                          dataloader1/dataloader2's batch_size.
    """

    def __init__(
        self,
        dataloader1: DataLoader,
        task_id_1: int,
        dataloader2: DataLoader,
        task_id_2: int,
        proportion_task1: float,
        batch_size: int,
    ):
        assert isinstance(task_id_1, int) and isinstance(task_id_2, int)
        if not (0.0 <= proportion_task1 <= 1.0):
            raise ValueError("proportion_task1 must be between 0.0 and 1.0")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        self.dataloader1 = dataloader1
        self.task_id_1 = task_id_1
        self.dataloader2 = dataloader2
        self.task_id_2 = task_id_2
        self.proportion_task1 = proportion_task1
        self.batch_size = batch_size

        self.current_shuffled_task_ids = []

        # Calculate the number of samples to take from each task for each combined batch
        self.num_samples_task1_per_batch = math.ceil(
            self.proportion_task1 * self.batch_size
        )
        self.num_samples_task2_per_batch = (
            self.batch_size - self.num_samples_task1_per_batch
        )

        self._length = min(len(dataloader1), len(dataloader2))

        if hasattr(dataloader1, "drop_last"):
            assert dataloader1.drop_last is True, "Drop last must be enabled."
        if hasattr(dataloader2, "drop_last"):
            assert dataloader2.drop_last is True, "Drop last must be enabled."

    def __len__(self):
        return self._length

    def __iter__(self):
        # Get iterators for both source dataloaders
        iter1 = iter(self.dataloader1)
        iter2 = iter(self.dataloader2)

        # Run until one of the dataloaders runs out
        for _ in range(len(self)):
            # Fetch a batch from each source dataloader
            batch1_data, batch1_labels = next(iter1)
            batch2_data, batch2_labels = next(iter2)

            # Select the required number of samples from each batch
            # We take a slice from the beginning of the batch, as they are already shuffled.
            selected_data1 = batch1_data[: self.num_samples_task1_per_batch]
            selected_labels1 = batch1_labels[: self.num_samples_task1_per_batch]

            selected_data2 = batch2_data[: self.num_samples_task2_per_batch]
            selected_labels2 = batch2_labels[: self.num_samples_task2_per_batch]

            # Concatenate data and labels from both tasks
            combined_data = torch.cat((selected_data1, selected_data2), dim=0)
            combined_labels = torch.cat((selected_labels1, selected_labels2), dim=0)

            # Shuffle the combined samples within the batch
            permutation = torch.randperm(self.batch_size)
            shuffled_data = combined_data[permutation]
            shuffled_labels = combined_labels[permutation]

            # Match up the task IDs
            task_ids_for_samples = torch.tensor(
                [self.task_id_1 for _ in range(self.num_samples_task1_per_batch)]
                + [self.task_id_2 for _ in range(self.num_samples_task2_per_batch)]
            )
            self.current_shuffled_task_ids = task_ids_for_samples[permutation]

            yield shuffled_data, shuffled_labels
