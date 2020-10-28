"""Module for training network"""

from typing import List

from modules.network.network import Autoencoder
from modules.data.base_dataset import BaseDataset

from tqdm import tqdm


def train_model(network: Autoencoder, dataset: BaseDataset, n_epochs: int) -> List[float]:
    tqdm_epochs = tqdm(range(n_epochs), postfix=f'Epochs...')

    total_error_list = list()

    for _ in tqdm_epochs:
        errors_epoch_list = list()

        for input_values, true_prediction in dataset:
            result = network.propagate_forward(x=input_values)

            # print(f'Model result: {result}')
            # print(f'True result: {true_prediction}')
            # print(f'-' * 5)

            error = network.propagate_backward(target=true_prediction)

            errors_epoch_list.append(error)

        average_error = sum(errors_epoch_list) / len(errors_epoch_list)

        tqdm_epochs.set_postfix(
            text=f'Epochs... Average error: {sum(errors_epoch_list) / len(errors_epoch_list):.2f}'
        )

        total_error_list.append(average_error)

    return total_error_list
