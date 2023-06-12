import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import numpy as np
from torch import flatten 
import matplotlib.pyplot as plt

def generate_mnist(portion=1, image_size=(28, 28)):
    # Define a transform to normalize the data, resize it and then flatten it
    transform = transforms.Compose([
        transforms.Resize(image_size),  # Resize images
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
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

def show_images(dataset, n=5):
    fig, axes = plt.subplots(1, n, figsize=(10, 2))
    for i in range(n):
        image, label = dataset[i]
        # Since the images are flattened, we reshape them back to their original shape
        image = image.view(28, 28)
        axes[i].imshow(image.numpy().squeeze(), cmap='gray')
        axes[i].set_title(label)
        axes[i].axis('off')
    plt.show()

# Generate the datasets
trainset, testset, validationset = generate_mnist()

# Show the first 5 images in the training set
show_images(trainset)