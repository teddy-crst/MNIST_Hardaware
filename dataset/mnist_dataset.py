import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import numpy as np
from torch import flatten 

def generate_mnist(portion=1.0):
    # Define a transform to normalize the data and flatten it
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(flatten)
    ])

    # Download the MNIST dataset
    dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, transform=transform)

    # Select a portion of the dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(portion * dataset_size))
    np.random.shuffle(indices)
    dataset_indices = indices[:split]
    dataset = torch.utils.data.Subset(dataset, dataset_indices)

    # Split the dataset into training, testing, and validation sets
    train_size = int(0.7 * len(dataset))
    test_size = int(0.15 * len(dataset))
    val_size = len(dataset) - train_size - test_size
    trainset, testset, validationset = random_split(dataset, [train_size, test_size, val_size])

    return trainset, testset, validationset
