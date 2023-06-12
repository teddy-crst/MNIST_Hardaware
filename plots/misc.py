import functools
from pathlib import Path
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import torch
from matplotlib.ticker import FuncFormatter

from utils.settings import settings

mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.linewidth'] = 2

mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['xtick.minor.width'] = 1.5

mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 5
mpl.rcParams['ytick.minor.width'] = 1.5


def plot_train_progress(loss_evolution: List[float], accuracy_evolution: List[dict] = None,
                        batch_per_epoch: int = 0) -> None:
    """
    Plot the evolution of the loss and the accuracy during the training.

    :param loss_evolution: A list of loss for each batch.
    :param accuracy_evolution: A list of dictionaries as {batch_num, test_accuracy, train_accuracy}.
    :param batch_per_epoch: The number of batch per epoch to plot x ticks.
    """
    with sns.axes_style("ticks"):
        fig, ax1 = plt.subplots()

        # Vertical lines for each batch
        if batch_per_epoch:
            for epoch in range(0, len(loss_evolution) + 1, batch_per_epoch):
                ax1.axvline(x=epoch, color='black', linestyle=':', alpha=0.2,
                            label='epoch' if epoch == 0 else '')  # only one label for the legend

        # Plot loss
        ax1.plot(loss_evolution, label='loss', color='tab:gray')
        ax1.set_ylabel('Loss')
        ax1.set_ylim(bottom=0)

        if accuracy_evolution:
            # Plot the accuracy evolution if available
            ax2 = plt.twinx()
            checkpoint_batches = [a['batch_num'] for a in accuracy_evolution]
            ax2.plot(checkpoint_batches, [a['test_accuracy'] for a in accuracy_evolution],
                     label='test accuracy',
                     color='tab:green')
            ax2.plot(checkpoint_batches, [a['train_accuracy'] for a in accuracy_evolution],
                     label='train accuracy',
                     color='tab:orange')
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            ax2.set_ylabel('Accuracy')
            ax2.set_ylim(bottom=0, top=1)

            # Place legends at the bottom
            ax1.legend(loc="lower left", bbox_to_anchor=(-0.1, -0.25))
            ax2.legend(loc="lower right", bbox_to_anchor=(1.2, -0.25))
        else:
            # Default legend position if there is only loss
            ax1.legend()

        plt.title('Training evolution')
        ax1.set_xlabel(f'Batch number (size: {settings.batch_size:n})')


#        save_plot('train_progress')

def plot_confusion_matrix(nb_labels_predictions: np.ndarray, class_names: List[str] = None,
                          annotations: bool = True) -> None:
    """
    Plot the confusion matrix for a set a predictions.

    :param nb_labels_predictions: The count of prediction for each label.
    :param class_names: The list of readable classes names
    :param annotations: If true the accuracy will be written in every cell
    """

    overall_accuracy = nb_labels_predictions.trace() / nb_labels_predictions.sum()
    rate_labels_predictions = nb_labels_predictions / nb_labels_predictions.sum(axis=1).reshape((-1, 1))
    plt.figure()
    sns.heatmap(rate_labels_predictions,
                vmin=0,
                vmax=1,
                square=True,
                fmt='.1%',
                cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto',
                annot=annotations,
                cbar=(not annotations))
    plt.title(f'Confusion matrix of {len(nb_labels_predictions)} classes '
              f'with {overall_accuracy * 100:.2f}% overall accuracy')
    plt.xlabel('Predictions')
    plt.ylabel('Labels')


def plot_fn(train, test, validation):
    """
    Basic plot function used for plotting the dataset
    Parameters
    ----------
    train: trainset
    test: testset
    validation: validationset

    Returns
    -------

    """
    train_coords, train_labels = train
    train_xs = train_coords[:, 0]
    train_ys = train_coords[:, 1]
    plt.scatter(train_xs, train_ys, s=1, label='train')
    if test is not None:
        test_coords, test_labels = test
        test_xs = test_coords[:, 0]
        test_ys = test_coords[:, 1]
        plt.scatter(test_xs, test_ys, s=1, label='test')
    if test is not None:
        validation_coords, validation_labels = validation
        validation_xs = validation_coords[:, 0]
        validation_ys = validation_coords[:, 1]
        plt.scatter(validation_xs, validation_ys, s=1, label='validation')
    plt.legend()
    plt.show()
    format_plot('$x$', '$f$')


def format_plot(x=None, y=None):
    if x is not None:
        plt.xlabel(x, fontsize=20)
    if y is not None:
        plt.ylabel(y, fontsize=20)


def finalize_plot(shape=(1, 1)):
    plt.gcf().set_size_inches(
        shape[0] * 1.5 * plt.gcf().get_size_inches()[1],
        shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
    plt.tight_layout()
    legend = functools.partial(plt.legend, fontsize=10)


def plot_weight_distribution(network):
    """
    Util function used to plot the distribution of the weights in the network
    Parameters
    ----------
    network: The Neural Network in question
    """
    all_weights = []
    for param in network.parameters():
        all_weights.extend(flatten_list(param.tolist()))
    all_weights = np.array(all_weights)
    mean = np.mean(all_weights)
    std = np.std(all_weights)
    fig = plt.figure()
    ax = fig.add_subplot()
    domain = np.linspace(mean - 3 * std, mean + 3 * std)
    ax.plot(domain, stats.norm.pdf(domain, mean, std), color="#1f77b4")
    ax.hist(all_weights, edgecolor='black', alpha=.5, bins=20, density=True, color="green")
    plt.grid()
    title = "Distribution of weights"
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability Density")
    plt.show()
    return


def flatten_list(_2d_list):
    """Flattens a list"""
    flat_list = []
    for element in _2d_list:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


def plot_uncertainty_single_image(image, network):
    """
    Plots the uncertainty in the predicted values for a single MNIST image.

    Parameters
    ----------
    image: The image to run through the network.
    network: The trained network to characterize.
    """
    # Run the image through the network 100 times to get a distribution of predictions
    outputs = network.infer(image.float().unsqueeze(0), 100)

    # Compute the standard deviation of the predictions
    std_dev = outputs[1][1].detach().numpy()

    # Compute the mean standard deviation if std_dev is a multi-valued numpy array
    if std_dev.size > 1:
        std_dev = np.mean(std_dev)

    # Plot the image
    plt.figure(figsize=(5, 5))
    image = image.reshape(28, 28)  # Reshape the image back to 2D
    plt.imshow(image, cmap='gray')
    plt.title(f'Average standard deviation of predictions: {std_dev:.4f}')
    plt.axis('off')
    plt.show()