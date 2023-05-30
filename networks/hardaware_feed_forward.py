from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tr
from torch import optim

from hardaware_pytorch_layers.hardaware_linear import Linear
from utils.settings import settings


class Hardaware_FeedForward(nn.Module):
    """
    Simple classifier neural network.
    Should be use as an example.
    """

    def __init__(self, input_size: 784, nb_classes: 10, elbo: bool):
        """
        Create a new network with 2 hidden layers fully connected.

        :param input_size: The size of one item of the dataset used for the training
        :param nb_classes: Number of class to classify
        """
        super().__init__()
        self.bayesian_nb_sample = settings.bayesian_nb_sample  # number of samples for the training if the bayesian averaging the loss approach is taken
        self.fc1 = Linear(input_size, settings.hidden_layers_size[0])  # Input -> Hidden 1
        self.dropout = nn.Dropout(settings.dropout)
        if nb_classes == 1:
            self.fc2 = Linear(settings.hidden_layers_size[0], 1)  # Hidden 2 -> Output
        else:
            self.fc2 = Linear(settings.hidden_layers_size[0], nb_classes, bias=True)  # Hidden 2 -> Output
        self.p_layers = [self.fc1.mask_w, self.fc1.mask_b, self.fc2.mask_w, self.fc2.mask_b]

        def weights_init(m):
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.elbo = elbo  # Boolean to determine if Elbo will be used or not
        self.fc1.apply(weights_init)
        self.fc2.apply(weights_init)
        self.layers = [self.fc1, self.fc2]
        if elbo:
            self._criterion = nn.CrossEntropyLoss(reduction='mean')  # reduction is sum since kld is computed in sum
        else:
            self._criterion = nn.CrossEntropyLoss(reduction='mean')
        self._optimizer = optim.Adam(self.parameters(), lr=settings.learning_rate, weight_decay=0)

    def forward(self, x: Any, training=False) -> Any:
        """
        Define the forward logic.

        :param x: One input of the dataset
        :return: The output of the network
        """
        x = self.dropout(x)
        x = torch.sigmoid(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def infer(self, inputs=784, nb_samples: int = 10):
        """
        Use network inference for classification a set of input.
        :param inputs: The inputs to classify.
        :param nb_sample: Not used here, just added for simple compatibility with Bayesian models.
        :return: The class inferred by this method and the confidence it this result (between 0 and 1).
        """
        # Prediction samples
        self.fc1.no_variability = False
        self.fc2.no_variability = False
        # Use sigmoid to convert the output into probability (during the training it's done inside BCEWithLogitsLoss)
        outputs_bef = [torch.sigmoid(self(inputs)) for _ in range(nb_samples)]
        outputs = torch.stack(outputs_bef)
        # Compute the mean, std

        all_equal = []
        for i in range(len(inputs)):
            with torch.no_grad():
                outputs_array = np.array(torch.round(outputs[:, i, 0]))
                all_equal.append(np.all(outputs_array == outputs_array[0]))
        means = outputs.mean(axis=0)
        stds = outputs.std(axis=0)
        # Round the samples mean value to 0 or 1
        predictions = torch.round(means)
        with torch.no_grad():
            individual_preds = np.round(np.array(outputs).squeeze())
        return predictions, (means, stds), individual_preds

    def training_step(self, inputs: Any, labels: Any):
        """
        Define the logic for one training step.

        :param inputs: The input from the training dataset, could be a batch or an item
        :param labels: The label of the item or the batch
        :return: The loss value
        """

        # Forward + Backward + Optimize
        losses = []
        loss = 0
        for i in range(self.bayesian_nb_sample):
            outputs = self(inputs, True)
            self.p_layers = [self.fc1.mask_w, self.fc1.mask_b, self.fc2.mask_w, self.fc2.mask_b]
            try:
                loss = self._criterion(outputs.squeeze(), labels)
            except:
                continue
            kld = 0  # Kullback Leibler Divergence term
            if self.elbo:
                for layer in self.layers:
                    kld += layer.kld
                elbo = loss + settings.bayesian_complexity_cost_weight * kld
                loss = elbo
            # Zero the parameter gradients
            losses.append(loss)
        if settings.bayesian_nb_sample > 1:
            for i in range(len(losses) - 1):
                loss += losses[i]
            loss = loss / settings.bayesian_nb_sample
        self._optimizer.zero_grad()
        if loss > 0:
            loss.backward()
        no_grad_values = []
        for layer in self.layers:
            no_grad_values.append([layer.weight[layer.mask_w_weight], layer.bias[layer.mask_w_bias]])
        self._optimizer.step()
        # ==== necessary to undo the gradient changes on substituted values ====
        i = 0
        with torch.no_grad():
            for layer in self.layers:
                layer.weight[layer.mask_w_weight] = no_grad_values[i][0].detach()
                layer.bias[layer.mask_w_bias] = no_grad_values[i][1].detach()
                i += 1
        return loss

    def get_hook(self, bit_map):
        def hook(grad):
            grad = grad.clone()  # NEVER change the given grad inplace
            # Assumes 1D but can be generalized
            grad[bit_map] = 0
            return grad

        return hook

    @staticmethod
    def get_transforms():
        """
        Define the data pre-processing to apply on the datasets before to use this neural network.
        """
        return tr.Compose([
            # Flatten the 28x28 image to a 784 array and convert to float
            tr.Lambda(lambda x: torch.flatten(x).float())
        ])

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
