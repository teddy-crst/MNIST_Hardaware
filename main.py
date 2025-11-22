import os
from pathlib import Path

import torch

from dataset.mnist_dataset import generate_mnist, preview_mnist_samples
from networks.feed_forward import FeedForward
from networks.hardaware_feed_forward import Hardaware_FeedForward
from plots.misc import plot_fn, plot_weight_distribution
from test_standard import test_standard
from train_standard import train_standard
from utils.logger import logger
from utils.settings import settings

network_dict = {1: Hardaware_FeedForward, 2: FeedForward}
name_network_dict = {1: "Hardaware", 2: "FeedForward"}
torch_device = torch.device("cpu")


def main():
    data_files = [
        settings.train_mnist_dataset_location,
        settings.test_mnist_dataset_location,
        settings.validation_mnist_dataset_location,
    ]
    os.makedirs(settings.dataset_dir, exist_ok=True)

    if settings.generate_new_mnist or not all(Path(path).exists() for path in data_files):
        if not settings.generate_new_mnist:
            logger.warning("Dataset artifacts missing; regenerating MNIST datasets.")
        trainset, testset, validationset = generate_mnist(root=settings.dataset_dir)
        torch.save(trainset, settings.train_mnist_dataset_location)
        torch.save(testset, settings.test_mnist_dataset_location)
        torch.save(validationset, settings.validation_mnist_dataset_location)
        preview_mnist_samples(trainset)
    else:  # Loading the dataset
        trainset = torch.load(settings.train_mnist_dataset_location)
        testset = torch.load(settings.test_mnist_dataset_location)
        validationset = torch.load(settings.validation_mnist_dataset_location)
    settings.bayesian_complexity_cost_weight = 1 / (trainset.__len__())
    logger.info("Selected network: " + name_network_dict[settings.choice])
    if settings.choice == 1:
        nn = network_dict[settings.choice](784, 10, elbo=settings.elbo)
    else:
        nn = network_dict[settings.choice](784, 10)
    train_standard(nn, trainset, testset, torch_device)
    acc = test_standard(nn, testset, torch_device)
    if settings.show_images or settings.save_images:
        plot_weight_distribution(nn)
    if settings.save_network:
        torch.save(nn, settings.pretrained_address)
    return


def compare_networks():
    """
    Method used to insert the weights of the regular Neural Network into the Hardware-Aware framework.
    Returns
    -------
    """
    nn = torch.load(os.getcwd() + "/good_trained_networks/FF_complete_precision_new_dataset/FF_166636344039526.pt")
    hann = torch.load(
        os.getcwd() + "/good_trained_networks/HAFF-complete_precision-P464E-Subs_5LRS_2HRS_AE_FIXED/HAFF_1667851816242543.pt")
    hann_cpy = torch.load(
        os.getcwd() + "/good_trained_networks/HAFF-complete_precision-P464E-Subs_5LRS_2HRS_AE_FIXED/HAFF_1667851816242543.pt")
    testset = torch.load(settings.test_mnist_dataset_location)
    with torch.no_grad():
        hann.fc1.weight = nn.fc1.weight
        hann.fc1.bias = nn.fc1.bias
        hann.fc2.weight = nn.fc2.weight
        hann.fc2.bias = nn.fc2.bias
    acc = test_standard(hann, testset, torch_device)
    acc = test_standard(hann_cpy, testset, torch_device)


if __name__ == '__main__':
    # cross_validate()
    main()
