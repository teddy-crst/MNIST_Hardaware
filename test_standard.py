import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from plots.misc import plot_uncertainty_predicted_value
from utils.logger import logger
from utils.settings import settings
from utils.timer import SectionTimer


def test_standard(network: Module, test_dataset: Dataset, device: torch.device, test_name: str = '',
                  final: bool = False,
                  limit: int = 0) -> float:
    """
    Start testing the network on a dataset.

    :param network: The network to use.
    :param test_dataset: The testing dataset.
    :param device: The device used to store the network and datasets (it can influence the behaviour of the testing)
    :param test_name: Name of this test for logging and timers.
    :param final: If true this is the final test, will show in log info and save results in file.
    :param limit: Limit of item from the dataset to evaluate during this testing (0 to run process the whole dataset).
    :return: The overall accuracy.
    """

    if test_name:
        test_name = ' ' + test_name

    nb_test_items = min(len(test_dataset), limit) if limit else len(test_dataset)
    logger.debug(f'Testing{test_name} on {nb_test_items:n} inputs')

    # Turn on the inference mode of the network
    network.eval()

    # Use the pyTorch data loader
    if test_name:
        test_name = ' ' + test_name

    nb_test_items = min(len(test_dataset), limit) if limit else len(test_dataset)
    logger.debug(f'Testing{test_name} on {nb_test_items:n} inputs')

    # Turn on the inference mode of the network
    network.eval()

    # Use the pyTorch data loader
    test_loader = DataLoader(test_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=0)
    nb_classes = 2  # len(test_dataset.classes) #TODO: CHECK HERE LEN TEST_DATASET
    nb_correct = 0  # counter of correct classifications
    nb_total = 0  # counter of total classifications
    # Create the tensors
    all_means = torch.Tensor()
    all_stds = torch.Tensor()
    all_inputs = torch.Tensor()
    all_outputs = torch.Tensor()
    # Disable gradient for performances
    with torch.no_grad(), SectionTimer(f'network testing{test_name}', 'info' if final else 'debug'):
        # Iterate batches
        for i, (inputs, labels) in enumerate(test_loader):
            # Stop testing after the limit
            if limit and i * settings.batch_size >= limit:
                break
            # Forward
            outputs = network.infer(inputs, settings.inference_number_contour)
            if settings.choice != 2:  # If Hardware-aware or Bayesian Network we need multiple inferences
                all_inputs = torch.cat((all_inputs, inputs))
                all_means = torch.cat((all_means, outputs[1][0]))
                all_stds = torch.cat((all_stds, outputs[1][1]))
                all_outputs = torch.cat((all_outputs, outputs[0]))
                nb_total += len(labels)
                nb_correct += int(torch.eq(outputs[0].flatten(), labels).sum())
            else:
                all_inputs = torch.cat((all_inputs, inputs))
                all_outputs = torch.cat((all_outputs, outputs.flatten()))
                nb_total += len(labels)
                nb_correct += int(torch.eq(outputs.flatten(), labels).sum())
    # accuracy
    accuracy = float(nb_correct / nb_total)
    plot_uncertainty_predicted_value(all_inputs, network)
    logger.info("mean accuracy: " + str(accuracy))
    return accuracy
