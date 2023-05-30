import argparse
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Sequence, Union

import configargparse
import torch.nn as nn
from numpy.distutils.misc_util import is_sequence

from utils.logger import logger


@dataclass(init=False, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class Settings:
    """
    Storing all settings for this program with default values.
    Setting are loaded from (last override first):
        - default values (in this file)
        - local file (default path: ./settings.yaml)
        - environment variables
        - arguments of the command line (with "--" in front)
    """

    # ==================================================================================================================
    # ==================================================== General =====================================================
    # ==================================================================================================================
    # The seed to use for all random number generator during this run.
    seed: int = 42
    logger_file_level = "INFO"

    # ==================================================================================================================
    # ============================================== Logging and Outputs ===============================================
    # ==================================================================================================================

    # The minimal logging level to show in the console (see https://docs.python.org/3/library/logging.html#levels).
    logger_console_level: Union[str, int] = 'INFO'

    # If True, use a visual progress bar in the console during training and loading.
    # Should be use with a logger_console_level as INFO or more for better output.
    visual_progress_bar: bool = True

    # If True show matplotlib images when they are ready.
    show_images: bool = True

    # If True and the run have a valid name, save matplotlib images in the run directory
    save_images: bool = True

    # If True and the run have a valid name, save the neural network parameters in the run directory at the end of the
    # training. Saved before applying early stopping if enabled.
    # The file will be at the root of run directory, under then name: "final_network.pt"
    save_network: bool = False
    # ==================================================================================================================
    # ==================================================== Dataset =====================================================
    # ==================================================================================================================
    mnist_input_size = 784
    mnist_nb_classes = 10
    # All neural network parameters:
    choice: int = 1
    # The number of training epoch.
    nb_epoch: int = 10000
    elbo: bool = False
    adj_sigma: bool = False
    bbyb: bool = False
    # The size of the mini-batch for the training and testing.
    batch_size: int = 256
    # The learning rate value used by the SGD for parameters update.
    learning_rate: float = 0.1
    HRS_failure_rate: float = 0.005
    LRS_failure_rate: float = 0.005
    # ==================================================================================================================

    # The percentage of data kept for testing only
    test_ratio: float = 0.1
    vmax = 0.4

    # The percentage of data kept for testing only
    validation_ratio: float = 0.1
    weight_clipping_scaler: float = 4.5
    # If True, data augmentation methods will be applied to increase the size of the train dataset.
    train_data_augmentation: bool = False  # currently unused

    generate_new_mnist = True
    inference_number_contour = 100
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    load_pretrained: bool = False
    overwrite_pretrained: bool = False
    overwrite_pretrained_bayesian: bool = True
    pretrained_address_dict = {1: (os.getcwd() + "/trained_networks/HAFF_" + str(timestamp).replace(".", "") + ".pt"),
                               2: (os.getcwd() + "/trained_networks/FF_" + str(timestamp).replace(".", "") + ".pt"),
                               3: (os.getcwd() + "/trained_networks/BFF_" + str(timestamp).replace(".", "") + ".pt")}
    pretrained_address = pretrained_address_dict[choice]
    train_mnist_dataset_location = './train_mnist_dataset.pt'
    test_mnist_dataset_location = './test_mnist_dataset.pt'
    validation_mnist_dataset_location = './validation_mnist_dataset.pt'
    # The number of data loader workers, to take advantage of multithreading. Always disable with CUDA.
    # 0 means automatic setting (using cpu count).
    nb_loader_workers: int = 0

    # If True, loss will compensate imbalance number of training examples between classes with weights.
    balance_with_weights: bool = False

    # ==================================================================================================================
    # ==================================================== Networks ====================================================
    # ==================================================================================================================

    # The number hidden layer and their respective number of neurons.
    hidden_layers_size: Sequence = (64,)

    # Dropout rate for every dropout layers defined in networks.
    # If a notwork model doesn't have a dropout layer this setting will have no effect.
    # 0 skip dropout layers
    dropout: int = 0.0

    # ==================================================================================================================
    # ==================================================== Training ====================================================
    # ==================================================================================================================

    # Prior distribution ratios for bayes by backprop
    prior_sigma1: float = 5
    prior_sigma2: float = 5
    posterior_rho_init: float = -3
    prior_pi: float = 0.6
    # The momentum value used by the SGD for parameters update.
    momentum: float = 0.9

    # criterion for half moon
    criterion = nn.BCEWithLogitsLoss()

    dataset_mu = 0  # currently unused
    dataset_sigma = 0  # currently unused

    train_points = 800
    # Save the best network state during the training based on the test accuracy.
    # Then load it when the training is complet.
    # The file will be at the root of run directory, under then name: "best_network.pt"
    # Required checkpoints_per_epoch > 0 and checkpoint_validation = True
    early_stopping: bool = False  # currently unused

    # The number of sample used to compute the loss of bayesian networks.
    bayesian_nb_sample: int = 1

    # The weight of complexity cost part when computing the loss of bayesian networks.
    bayesian_complexity_cost_weight: float = (1 / train_points) * (1 / batch_size)  # 1. / train_points

    # ==================================================================================================================
    # ================================================== Checkpoints ===================================================
    # ==================================================================================================================

    # The number of checkpoints per training epoch, if 0 no checkpoint is processed
    checkpoints_per_epoch: int = 0

    # The number of data in the checkpoint training subset.
    # Set to 0 to don't compute the train accuracy during checkpoints.
    checkpoint_train_size: int = 0
    checkpoint_test_size: int = 50

    # If the inference accuracy of the validation dataset should be computed, or not, during checkpoint.
    checkpoint_validation: bool = True

    # If True and the run have a valid name, save the neural network parameters in the run directory at each checkpoint.
    checkpoint_save_network: bool = False

    def validate(self):
        """
        Validate settings.
        """
        # Logging and Outputs
        possible_log_levels = ('CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET')
        assert self.logger_console_level.upper() in possible_log_levels or isinstance(self.logger_console_level, int), \
            f"Invalid console log level '{self.logger_console_level}'"
        assert self.logger_file_level.upper() in possible_log_levels or isinstance(self.logger_file_level, int), \
            f"Invalid file log level '{self.logger_file_level}'"

        # Dataset
        assert self.test_ratio > 0, 'Test data ratio should be more than 0'
        assert self.test_ratio + self.validation_ratio < 1, 'test_ratio + validation_ratio should be less than 1 to' \
                                                            ' have training data'

        # Networks
        assert all((a > 0 for a in self.hidden_layers_size)), 'Hidden layer size should be more than 0'

        # Training
        assert self.batch_size > 0, 'Batch size should be a positive integer'
        assert self.nb_epoch > 0, 'Number of epoch should be at least 1'
        assert self.bayesian_nb_sample > 0, 'The number of bayesian sample should be at least 1'

        # Checkpoints
        assert self.checkpoints_per_epoch >= 0, 'The number of checkpoints should be >= 0'

    def __init__(self):
        """
        Create the setting object.
        """
        self._load_file_and_cmd()

    def _load_file_and_cmd(self) -> None:
        """
        Load settings from local file and arguments of the command line.
        """

        def str_to_bool(arg_value: str) -> bool:
            """
            Used to handle boolean settings.
            If not the 'bool' type convert all not empty string as true.
            :param arg_value: The boolean value as a string.
            :return: The value parsed as a string.
            """
            if isinstance(arg_value, bool):
                return arg_value
            if arg_value.lower() in {'false', 'f', '0', 'no', 'n'}:
                return False
            elif arg_value.lower() in {'true', 't', '1', 'yes', 'y'}:
                return True
            raise argparse.ArgumentTypeError(f'{arg_value} is not a valid boolean value')

        def type_mapping(arg_value):
            if type(arg_value) == bool:
                return str_to_bool
            if is_sequence(arg_value):
                if len(arg_value) == 0:
                    return str
                else:
                    return type_mapping(arg_value[0])

            # Default same as current value
            return type(arg_value)

        p = configargparse.get_argument_parser(default_config_files=['./settings.yaml'])

        # Spacial argument
        p.add_argument('-s', '--settings', required=False, is_config_file=True,
                       help='path to custom configuration file')

        # Create argument for each attribute of this class
        for name, value in asdict(self).items():
            p.add_argument(f'--{name.replace("_", "-")}',
                           f'--{name}',
                           dest=name,
                           required=False,
                           action='append' if is_sequence(value) else 'store',
                           type=type_mapping(value))

        # Load arguments form file, environment and command line to override the defaults
        for name, value in vars(p.parse_args()).items():
            if name == 'settings':
                continue
            if value is not None:
                # Directly set the value to bypass the "__setattr__" function
                self.__dict__[name] = value

        self.validate()

    def __setattr__(self, name, value) -> None:
        """
        Set an attribute and valide the new value.
        :param name: The name of the attribut
        :param value: The value of the attribut
        """
        logger.debug(f'Setting "{name}" changed from "{getattr(self, name)}" to "{value}".')
        self.__dict__[name] = value

    def __delattr__(self, name):
        raise AttributeError('Removing a setting is forbidden for the sake of consistency.')

    def __str__(self) -> str:
        """
        :return: Human readable description of the settings.
        """
        return 'Settings:\n\t' + \
            '\n\t'.join([f'{name}: {str(value)}' for name, value in asdict(self).items()])


# Singleton setting object
settings = Settings()
