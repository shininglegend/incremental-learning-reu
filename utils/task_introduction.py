# Written by Abigail Dodd
# Helper file for deciding task order
import math
import random
import torch
from torch.utils.data import DataLoader
from dataset_tools.load_dataset import DatasetLoader


def get_epoch_order(train_dataloaders: list, config: dict):
    '''
    Accepts configuration parameters
    Expected config keys: 'task_introduction', 'num_tasks', 'num_epochs'

    Currently supports:
    'sequential' - 20 epochs per task, introduced in sequence
    'half and half' - 1/2 of total epochs per task in sequence, then 1/2 total epochs again in task sequence
    'random' - num_epochs per task total, but the program will decide the order.
    '''

    if config['task_introduction'] == ('continual' or 'continuous'):
        return continual_change(train_dataloaders, config=config)

    task_epochs_dict = _make_num_epochs_into_dict(config)
    train_dataloaders_dict = {index: value for index, value in enumerate(train_dataloaders)}

    task_introduction = config['task_introduction']

    if task_introduction == 'sequential':
        return _get_pure_sequential_order(task_epochs_dict), train_dataloaders_dict
    elif task_introduction == 'half and half':
        return _get_half_and_half_epoch_order(task_epochs_dict), train_dataloaders_dict
    elif task_introduction == 'random':
        return _get_random_epoch_order(task_epochs_dict), train_dataloaders_dict
    else:
        print(f"config[\'task_introduction\'] has value \"{config['task_introduction']}\", which isn't currently supported.")
        print("Please edit default.yaml or task_introduction.py and try again.")
        exit("Goodbye. :)")


def _get_random_epoch_order(task_epochs_dict):
    epoch_order = _get_pure_sequential_order(task_epochs_dict)
    random.shuffle(epoch_order)
    print("ts random i promise")
    return epoch_order


def _get_half_and_half_epoch_order(task_epochs_dict):
    '''
    Inherently assumes config['num_epochs'] is a dictionary.
    '''
    epoch_order = []
    first_half = []
    second_half = []

    for task_id, epoch_num in enumerate(task_epochs_dict):
        for i in range(math.ceil(epoch_num / 2)):
            first_half.append(task_id)
        for i in range(math.floor(epoch_num / 2)):
            second_half.append(task_id)

    epoch_order = first_half + second_half
    return epoch_order


def _get_pure_sequential_order(task_epochs_dict):
    '''
    Takes in: dict of key:value where key is task_id and value is number of epochs for that id
    Returns a list that contains the id of the task, in the order that we want to do them.
    '''

    epoch_order = []

    for task_id, num_epochs in enumerate(task_epochs_dict):
        for i in range(num_epochs):
            epoch_order.append(task_id)

    return epoch_order


def _make_num_epochs_into_dict(config: dict):
    """
    Checks the type of num_epochs_per_task and performs the necessary functions.
    """
    num_epochs_per_task = config["num_epochs"]

    task_nums_and_epochs_dict = {}

    if isinstance(num_epochs_per_task, list):
        task_nums_and_epochs_dict = dict(enumerate(num_epochs_per_task))
    elif isinstance(num_epochs_per_task, int):
        # Assume every task is the name number of epochs.
        for i in range(config['num_tasks']):
            task_nums_and_epochs_dict[i] = num_epochs_per_task
    elif isinstance(num_epochs_per_task, dict):
        task_nums_and_epochs_dict = num_epochs_per_task

    config['num_epochs'] = task_nums_and_epochs_dict
    return task_nums_and_epochs_dict


def continual_change(train_dataloaders: list[DataLoader], config: dict):
    '''
    Takes in num_tasks train_dataloaders and alters them for continual learning.
    See "section F.5. Continuous change experiments" in the Lamers appendix for more details.

    To better match lamers, we assume that continual change has an "overlap" of 10 epochs.
    So we alternate: 10 pure, 10 mixed, 10 pure, 10 mixed, ..... 10 pure, end with 10 final task to initial task mixed.
    '''

    num_transition_epochs = config['num_transition_epochs']

    assert num_transition_epochs is int, "num_transition_epochs must be an integer for continuous change."
    assert num_transition_epochs is not None, "num_transition_epochs doesn't exist. Try again."
    if num_transition_epochs <= 0:
        print(f"num_transition_epochs <= 0. That won't work for us, bestie.")
        print("We're gonna guess that you meant to run a sequential task introduction vibe.")
        print("Don't worry, we'll fix this little hiccup and you'll be on your merry way.")
        config['task_introduction'] = 'sequential, but they messed up'
        task_epochs_dict = _make_num_epochs_into_dict(config)
        return _get_pure_sequential_order(task_epochs_dict), train_dataloaders

    # Now we can assume num_transition_epochs is a positive integer
    step = 1/num_transition_epochs
    mixed_dataloaders_dict = _get_continual_change_dataloaders(train_dataloaders, step, config)

    mixed_epoch_list = []
    epochs_per_task = config['num_epochs']
    config['num_epochs'] = {}

    for task_index in range(config['num_tasks']):
        # Here, we assume num_epochs is equal for all tasks.
        for i in range(epochs_per_task - num_transition_epochs):
            mixed_epoch_list.append(task_index)
        for i in range(num_transition_epochs):
            mixed_epoch_list.append(task_index + i * step)
        config['num_epochs'][task_index] = epochs_per_task + num_transition_epochs

    return mixed_epoch_list, mixed_dataloaders_dict


def _get_continual_change_dataloaders(train_dataloaders: list[DataLoader], step, config):
    mixed_dataloaders_dict = {}
    for task_id, task_dataloader in enumerate(train_dataloaders):
        p = 1.0 - step
        mixed_dataloaders_dict[task_id] = task_dataloader

        next_task_id = task_id + 1
        if next_task_id >= len(train_dataloaders):
            next_task_id = -1

        current_dataloader = train_dataloaders[task_id]
        next_dataloader = train_dataloaders[next_task_id]

        while p > 0:
            mixed_dataloaders_dict[task_id + p] = MixedTaskDataLoader(
                current_dataloader,
                next_dataloader,
                proportion_task1=1-p,
                batch_size=config['batch_size'],
                config=config
            )
            p -= step

    return mixed_dataloaders_dict


class MixedTaskDataLoader:
    """
        A custom dataloader that combines batches from two source dataloaders
        while maintaining a specified proportion of samples from each.
        The combined samples within each batch are shuffled.

        Args:
            dataloader1 (torch.utils.data.DataLoader): DataLoader for Task 1.
            dataloader2 (torch.utils.data.DataLoader): DataLoader for Task 2.
            proportion_task1 (float): The desired proportion of samples from Task 1
                                      in each combined batch (0.0 to 1.0).
                                      A proportion of 1.0 means all samples will be from Task 1.
            batch_size (int): The batch size for the combined batches.
                              This should match the batch sizes of dataloader1
                              and dataloader2, or be a multiple/divisor of them.
                              For simplicity, it's assumed to be the same as
                              dataloader1/dataloader2's batch_size.
            config (dict): configuration dictionary with miscellaneous parameters.
        """

    def __init__(self,
                 dataloader1: DataLoader,
                 dataloader2: DataLoader,
                 proportion_task1: float,
                 batch_size: int,
                 config: dict):
        if not (0.0 <= proportion_task1 <= 1.0):
            raise ValueError("proportion_task1 must be between 0.0 and 1.0")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
        self.proportion_task1 = proportion_task1
        self.batch_size = batch_size
        self.config = config

        # Calculate the number of samples to take from each task for each combined batch
        self.num_samples_task1_per_batch = math.ceil(self.proportion_task1 * self.batch_size)
        self.num_samples_task2_per_batch = self.batch_size - self.num_samples_task1_per_batch

        # Ensure that the underlying dataloaders have drop_last=True for consistent batch sizes
        # This is crucial for guaranteeing exact proportions in every batch.
        if hasattr(dataloader1, 'drop_last') and not dataloader1.drop_last:
            print("Warning: dataloader1 does not have drop_last=True. Last batch might be incomplete.")
        if hasattr(dataloader2, 'drop_last') and not dataloader2.drop_last:
            print("Warning: dataloader2 does not have drop_last=True. Last batch might be incomplete.")

        self._length = self.config['']

    def __len__(self):
        return self._length

    def __iter__(self):
        # Get iterators for both source dataloaders
        iter1 = iter(self.dataloader1)
        iter2 = iter(self.dataloader2)

        while True:
            try:
                # Fetch a batch from each source dataloader
                batch1_data, batch1_labels = next(iter1)
                batch2_data, batch2_labels = next(iter2)

                # Ensure we have enough samples in the fetched batches
                if batch1_data.size(0) < self.num_samples_task1_per_batch:
                    print(
                        f"Warning: Task 1 batch size ({batch1_data.size(0)}) is less than required ({self.num_samples_task1_per_batch}). Stopping iteration.")
                    break
                if batch2_data.size(0) < self.num_samples_task2_per_batch:
                    print(
                        f"Warning: Task 2 batch size ({batch2_data.size(0)}) is less than required ({self.num_samples_task2_per_batch}). Stopping iteration.")
                    break

                # Select the required number of samples from each batch
                # We take a slice from the beginning of the batch.
                # If you need random samples from within the source batch,
                # you'd need to shuffle/sample from batch1_data/labels first.
                selected_data1 = batch1_data[:self.num_samples_task1_per_batch]
                selected_labels1 = batch1_labels[:self.num_samples_task1_per_batch]

                selected_data2 = batch2_data[:self.num_samples_task2_per_batch]
                selected_labels2 = batch2_labels[:self.num_samples_task2_per_batch]

                # Concatenate data and labels from both tasks
                combined_data = torch.cat((selected_data1, selected_data2), dim=0)
                combined_labels = torch.cat((selected_labels1, selected_labels2), dim=0)

                # Shuffle the combined samples within the batch
                permutation = torch.randperm(self.batch_size)
                shuffled_data = combined_data[permutation]
                shuffled_labels = combined_labels[permutation]

                yield shuffled_data, shuffled_labels

            except StopIteration:
                # One of the dataloaders has run out of batches
                print("One of the underlying dataloaders exhausted. Stopping mixed iteration.")
                break