from typing import List

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from plots.misc import plot_train_progress
from test_standard import test_standard
from utils.logger import logger
from utils.settings import settings
from utils.timer import SectionTimer


def train_standard(network: Module, train_dataset: Dataset, test_dataset: Dataset, device: torch.device) -> None:
    """
    Train the network using the dataset.

    :param network: The network to train in place.
    :param train_dataset: The dataset used to train the network.
    :param test_dataset: The dataset used to run intermediate test on the network during the training.
    :param device: The device used to store the network and datasets (it can influence the behaviour of the training)
    """
    network.train()

    # Use the pyTorch data loader
    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=0)
    nb_batch = len(train_loader)
    # Define the indexes of checkpoints for each epoch
    # Eg.: with 'nb_batch' = 100 and 'checkpoints_per_epoch' = 3, then the indexes will be [0, 33, 66]
    checkpoints_i = [int(i / settings.checkpoints_per_epoch * nb_batch) for i in range(settings.checkpoints_per_epoch)]
    # Store the loss values for plot
    loss_evolution: List[float] = []
    accuracy_evolution: List[dict] = []
    epochs_stats: List[dict] = []
    with SectionTimer('network training') as timer:
        # Iterate epoch
        for epoch in range(settings.nb_epoch):
            # Iterate batches
            for i, (inputs, labels) in enumerate(train_loader):

                if i in checkpoints_i:
                    timer.pause()
                    accuracy_evolution.append(
                        _checkpoint(network, epoch * nb_batch + i, train_dataset, test_dataset, device))
                    timer.resume()

                # Run a training set for these data
                loss = network.training_step(inputs, labels)
                loss_evolution.append(float(loss))
                if type(network) is None:  # Hardaware_FeedForward): #TODO: VERIFY IF WE GET HERE
                    with torch.no_grad():
                        all_weights = []
                        for param in network.parameters():
                            all_weights.extend(flatten_list(param.tolist()))
                        all_weights = np.array(all_weights)
                        mean = np.mean(all_weights)
                        std = np.std(all_weights)
                        upper_bound = mean + settings.weight_clipping_scaler * std
                        lower_bound = mean - settings.weight_clipping_scaler * std
                        for param in network.parameters():
                            param.clamp_(lower_bound, upper_bound)

            # Epoch statistics
            _record_epoch_stats(epochs_stats, loss_evolution[-len(train_loader):])
        plot_train_progress(loss_evolution, accuracy_evolution)


def flatten_list(_2d_list):
    flat_list = []
    for element in _2d_list:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


def _checkpoint(network: Module, batch_num: int, train_dataset: Dataset, test_dataset: Dataset, device) -> dict:
    """
    Pause the training to do some jobs, like intermediate testing and network backup.

    :param network: The current neural network
    :param batch_num: The batch number since the beginning of the training (used as checkpoint id)
    :param train_dataset: The training dataset
    :param test_dataset: The testing dataset
    :return: A dictionary with the tests results
    """

    # Save the current network

    # Start tests
    test_accuracy = test_standard(network, test_dataset, test_name='checkpoint test',
                                  limit=settings.checkpoint_test_size,
                                  device=device)
    train_accuracy = test_standard(network, train_dataset, test_name='checkpoint train',
                                   limit=settings.checkpoint_train_size,
                                   device=device)
    # Set it back to train because it was switched during tests
    network.train()

    logger.info(f'Checkpoint {batch_num:<6n} '
                f'| test accuracy: {test_accuracy:5.2%} '
                f'| train accuracy: {train_accuracy:5.2%}')

    return {
        'batch_num': batch_num,
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy
    }


def _record_epoch_stats(epochs_stats: List[dict], epoch_losses: List[float]) -> None:
    """
    Record the statics for one epoch.

    :param epochs_stats: The list where to store the stats, append in place.
    :param epoch_losses: The losses list of the current epoch.
    """
    stats = {
        'losses_mean': float(np.mean(epoch_losses)),
        'losses_std': float(np.std(epoch_losses))
    }

    # Compute the loss difference with the previous epoch
    stats['losses_mean_diff'] = 0 if len(epochs_stats) == 0 else stats['losses_mean'] - epochs_stats[-1]['losses_mean']

    # Add stat to the list
    epochs_stats.append(stats)

    # Log stats
    epoch_num = len(epochs_stats)
    logger.info(f"Epoch {epoch_num:3}/{settings.nb_epoch} ({epoch_num / settings.nb_epoch:7.2%}) "
                f"| loss: {stats['losses_mean']:.5f} "
                f"| diff: {stats['losses_mean_diff']:+.5f} "
                f"| std: {stats['losses_std']:.5f}")
