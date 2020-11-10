"""Module for training network"""

from typing import List, Optional

from modules.network.network import Autoencoder
from modules.data.base_dataset import BaseDataset

from tqdm import tqdm


def train_model(network: Autoencoder, dataset: BaseDataset,
                n_epochs: int, min_error: Optional[float] = None,
                verbose: bool = True) -> List[float]:
    """
    Training method for network

    :param network: neural network
    :param dataset: dataset
    :param n_epochs: number of epochs to train
    :param min_error: minimal error we need to achieve, if None no threshold
    :param verbose: if True shows progress bar
    :return: list of errors for give all epoch
    """

    n_epochs_iter = range(n_epochs)

    if verbose:
        n_epochs_iter = tqdm(n_epochs_iter, postfix=f'Epochs...')

    total_error_list = list()

    for _ in n_epochs_iter:
        errors_epoch_list = list()

        for input_values, true_prediction in dataset:
            result = network.propagate_forward(x=input_values)
            error = network.propagate_backward(target=true_prediction)

            errors_epoch_list.append(error)

        average_error = sum(errors_epoch_list) / len(errors_epoch_list)

        if min_error and average_error < min_error:
            return total_error_list

        if verbose:
            n_epochs_iter.set_postfix(
                text=f'Epochs... Average error: {sum(errors_epoch_list) / len(errors_epoch_list):.4f}'
            )

        total_error_list.append(average_error)

    return total_error_list

