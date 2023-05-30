from typing import Any

import torch
import torch.nn as nn
from torch import optim

from utils.settings import settings


class FeedForward(nn.Module):
    """
    Simple classifier neural network.
    """

    def __init__(self, input_size: int, nb_classes: int):
        """
        Create a new network with 2 hidden layers fully connected.

        :param input_size: The size of one item of the dataset used for the training
        :param nb_classes: Number of class to classify
        """
        super().__init__()

        self.fc1 = nn.Linear(input_size, settings.hidden_layers_size[0])  # Input -> Hidden 1
        if nb_classes == 1:
            self.fc2 = nn.Linear(settings.hidden_layers_size[0], 1)  # Hidden 2 -> Output
        else:
            self.fc2 = nn.Linear(settings.hidden_layers_size[0], nb_classes, bias=False)  # Hidden 2 -> Output

        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.fc1.apply(weights_init)
        self.fc2.apply(weights_init)
        self._criterion = nn.BCEWithLogitsLoss()
        self._optimizer = optim.Adam(self.parameters(), lr=settings.learning_rate)

    def forward(self, x: Any) -> Any:
        """
        Define the forward logic.

        :param x: One input of the dataset
        :return: The output of the network
        """
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

    def infer(self, inputs):
        """
        Use network inference for classification a set of input.
        :param inputs: The inputs to classify.
        :return: The class inferred by this method and the confidence it this result (between 0 and 1).
        """
        # Use sigmoid to convert the output into probability (during the training it's done inside BCEWithLogitsLoss)
        outputs = torch.sigmoid(self(inputs))

        # We assume that a value far from 0 or 1 mean low confidence (e.g. output:0.25 => class 0 with 50% confidence)
        predictions = torch.round(outputs)  # Round to 0 or 1
        return predictions

    def training_step(self, inputs: Any, labels: Any):
        """
        Define the logic for one training step.

        :param inputs: The input from the training dataset, could be a batch or an item
        :param labels: The label of the item or the batch
        :return: The loss value
        """
        # Zero the parameter gradients
        self._optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = self(inputs)
        loss = self._criterion(outputs.squeeze(), labels)
        loss.backward()
        self._optimizer.step()
        return loss

    def get_loss_name(self) -> str:
        """
        :return: The name of the loss function (criterion).
        """
        return type(self._criterion).__name__

    def get_optimizer_name(self) -> str:
        """
        :return: The name of the optimiser function.
        """
        return type(self._optimizer).__name__
