import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

# Define a top-level function for flattening the input
def flatten(x):
    return torch.flatten(x)

def generate_mnist():
    # Define a transform to normalize the data and flatten it
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(flatten)
    ])

    # Download the MNIST dataset
    dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, transform=transform)

    # Split the dataset into training, testing, and validation sets
    train_size = int(0.7 * len(dataset))
    test_size = int(0.15 * len(dataset))
    val_size = len(dataset) - train_size - test_size
    trainset, testset, validationset = random_split(dataset, [train_size, test_size, val_size])

    return trainset, testset, validationset
