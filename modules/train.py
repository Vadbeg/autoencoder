"""Module for training network"""

from typing import List

from modules.network.network import Autoencoder
from modules.data.base_dataset import BaseDataset

from tqdm import tqdm


def train_model(network: Autoencoder, dataset: BaseDataset, n_epochs: int) -> List[float]:
    """
    Training method for network

    :param network: neural network
    :param dataset: dataset
    :param n_epochs: number of epochs to train
    :return: list of errors for give all epoch
    """

    tqdm_epochs = tqdm(range(n_epochs), postfix=f'Epochs...')

    total_error_list = list()

    for _ in tqdm_epochs:
        errors_epoch_list = list()

        for input_values, true_prediction in dataset:
            result = network.propagate_forward(x=input_values)
            error = network.propagate_backward(target=true_prediction)

            # print(f'Result: {result}')
            # print(f'-' * 15)

            errors_epoch_list.append(error)

        average_error = sum(errors_epoch_list) / len(errors_epoch_list)

        # if average_error < threshold:
        #     return total_error_list

        tqdm_epochs.set_postfix(
            text=f'Epochs... Average error: {sum(errors_epoch_list) / len(errors_epoch_list):.2f}'
        )

        total_error_list.append(average_error)

    return total_error_list

