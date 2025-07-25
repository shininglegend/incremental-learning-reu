# Written by Abigail Dodd
# Helper file for deciding task order
import math
import random


def get_epoch_order(config: dict):
    '''
    Accepts configuration parameters
    one parameter should be 'task_introduction', and another should be 'num_epochs'

    Currently supports:
    'sequential' - 20 epochs per task, introduced in sequence
    'half and half' - 1/2 of total epochs per task in sequence, then 1/2 total epochs again in task sequence
    'random' - num_epochs per task total, but the program will decide the order.
    '''

    check_num_epochs_type(config['num_epochs'])
    # should be a dictionary
    # Key-value pairs should be task_index, number epochs

    task_introduction = config['task_introduction']

    if task_introduction == 'sequential':
        return get_pure_sequential_order(config)
    elif task_introduction == 'half and half':
        return get_half_and_half_epoch_order(config)
    elif task_introduction == 'random':
        return get_random_epoch_order(config)
    else:
        print(f"config[\'task_introduction\'] has value \"{config['task_introduction']}\", which isn't currently supported.")
        print("Please edit default.yaml or task_introduction.py and try again.")
        exit("Goodbye. :)")


def get_random_epoch_order(config: dict):
    epoch_order = get_pure_sequential_order(config)
    random.shuffle(epoch_order)
    return epoch_order


def get_half_and_half_epoch_order(config: dict):
    '''
    Inherently assumes config['num_epochs'] is a dictionary.
    '''
    epoch_order = []

    for task_id in range(config['num_tasks']):
        for i in range(math.ceil(config['num_epochs'][task_id] / 2)):
            epoch_order.append(task_id)

    for task_id in range(config['num_tasks']):
        for i in range(math.floor(config['num_epochs'][task_id] / 2)):
            epoch_order.append(task_id)

    # debug
    assert len(epoch_order) == config['total_epochs'], "get_half_and_half error in " \
                                                                           "task_introduction.py "

    return epoch_order


def get_pure_sequential_order(config: dict):
    '''
    For now, assumes each task has the same number of epochs.
    If we want to implement variable numbers of epochs, we can do that later using lists or dicts

    '''

    epoch_order = []

    for task_id, num_epochs in enumerate(config['num_epochs']):
        for i in range(num_epochs):
            epoch_order.append(task_id)

    # debug
    assert len(epoch_order) == config['total_epochs'], \
        "get_pure_sequential error in task_introduction.py"

    return epoch_order


def check_num_epochs_type(num_epochs_per_task):
    '''
    Checks the type of num_epochs_per_task and performs the necessary functions.

    Long term, I would like for this file to support a dictionary for num_epochs_per_task,
    where the keys are task IDs and the values are the number of epochs per task.

    wait why don't i juse use a list.
    '''

    if isinstance(num_epochs_per_task, list):
        print("You gave a list for epochs per task.")
        print("While this could be supported in theory (with clearly numbered tasks as indices), it hasn't been implemented yet.")
        print("See task_introduction.py, check_num_epochs_type, for more details.")
        exit("Goodbye.")

    assert isinstance(num_epochs_per_task, dict), "config['num_epochs'] should be an dict."
