import torch
from torch.utils.data import DataLoader, TensorDataset
import random


def mix_train_dataloaders(train_dataloaders, config):
    mixed_loaders = []

    num_tasks = config['num_tasks']
    batch_size = config['batch_size']

    for task_id in range(num_tasks):
        # Wraparound to next task
        next_task_id = (task_id + 1) % num_tasks

        # Grab full datasets from both tasks
        x_a, y_a = train_dataloaders[task_id].dataset.tensors
        x_b, y_b = train_dataloaders[next_task_id].dataset.tensors

        def make_task_mixer(x_a, y_a, x_b, y_b):
            def mixed_loader(q):
                # Shuffle both datasets independently each time
                idx_a = torch.randperm(len(x_a))
                idx_b = torch.randperm(len(x_b))
                x_a_shuf, y_a_shuf = x_a[idx_a], y_a[idx_a]
                x_b_shuf, y_b_shuf = x_b[idx_b], y_b[idx_b]

                # Create iterators
                i_a, i_b = 0, 0
                total_samples = len(x_a)
                batches = total_samples // batch_size

                for _ in range(batches):
                    n_a = int(q * batch_size)
                    n_b = batch_size - n_a

                    x_batch = torch.cat([x_a_shuf[i_a:i_a + n_a], x_b_shuf[i_b:i_b + n_b]], dim=0)
                    y_batch = torch.cat([y_a_shuf[i_a:i_a + n_a], y_b_shuf[i_b:i_b + n_b]], dim=0)

                    i_a += n_a
                    i_b += n_b

                    # Shuffle the batch so samples are mixed
                    idx = torch.randperm(len(x_batch))
                    yield x_batch[idx], y_batch[idx]
            return mixed_loader

        mixed_loaders.append(make_task_mixer(x_a, y_a, x_b, y_b))

    return mixed_loaders

if __name__ == "__main__":
    print("we running")