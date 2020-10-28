"""Module for training setup"""

import math
from typing import Tuple

from modules.train import train_model
from modules.utils import calculate_image_compression
from modules.network.network import Autoencoder
from modules.network.utils import sigmoid
from modules.data.dataset import ImageAutoencoderDataset
from config import Config

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from cv2 import cv2


def input_part_into_res_image(res_image: np.ndarray, image_chunk: np.ndarray,
                              idx: int, slide_window: Tuple[int, int]):
    y_window, x_window = slide_window

    max_x = math.ceil(res_image.shape[1] / x_window)
    max_y = math.ceil(res_image.shape[0] / y_window)

    y_idx = idx // max_x
    x_idx = idx % max_x

    assert x_idx < max_x, 'X_idx is more than max X'
    assert x_idx >= 0, 'X_idx is less than zero'

    assert y_idx < max_y, 'Y_idx is more than max Y'
    assert y_idx >= 0, 'Y_idx is less than zero'

    x_idx = x_idx * x_window
    y_idx = y_idx * y_window

    if y_idx + y_window > res_image.shape[0]:
        y_idx = res_image.shape[0] - y_window

    if x_idx + x_window > res_image.shape[1]:
        x_idx = res_image.shape[1] - x_window

    res_image[y_idx: y_idx + y_window, x_idx: x_idx + x_window] = image_chunk

    return res_image


def perform_pipeline():
    y_length, x_length = Config.slide_window
    num_of_nodes = y_length * x_length * 3

    autoencoder = Autoencoder(lr=Config.learning_rate, momentum=0.1, adaptive_lr=Config.adaptive_lr,
                              shape=[num_of_nodes, Config.num_of_hidden_layers, num_of_nodes])

    dataset = ImageAutoencoderDataset(image_path=Config.image_path,
                                      image_size=Config.image_size,
                                      slide_window=Config.slide_window)

    total_error_list = train_model(network=autoencoder, dataset=dataset, n_epochs=Config.n_epochs)

    res_image = np.zeros(dataset.image.shape)
    true_image = dataset.image

    for idx, (input_image_flatten, true_image_flatten) in enumerate(dataset):
        result = autoencoder.propagate_forward(x=input_image_flatten)
        result = sigmoid(result)

        result = result.reshape((*Config.slide_window, 3))

        res_image = input_part_into_res_image(res_image=res_image, image_chunk=result,
                                              idx=idx, slide_window=Config.slide_window)

    res_image = np.uint8(res_image * 255)

    res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)
    true_image = cv2.cvtColor(true_image, cv2.COLOR_BGR2RGB)

    compression_rate = calculate_image_compression(num_of_input_layers=num_of_nodes,
                                                   num_of_chunks=len(dataset),
                                                   num_of_hidden_layers=Config.num_of_hidden_layers)

    return res_image, true_image, total_error_list, compression_rate


def iterations_depending_on_learning():
    lr_list = [0.0001, 0.0003, 0.001, 0.0075, 0.01, 0.015, 0.03, 0.05, 0.09]
    epochs_list = list()

    from tqdm import tqdm

    for curr_learning in tqdm(lr_list):
        y_length, x_length = Config.slide_window
        num_of_nodes = y_length * x_length * 3

        autoencoder = Autoencoder(lr=curr_learning, momentum=0.1, adaptive_lr=Config.adaptive_lr,
                                  shape=[num_of_nodes, Config.num_of_hidden_layers, num_of_nodes])

        dataset = ImageAutoencoderDataset(image_path=Config.image_path,
                                          image_size=Config.image_size,
                                          slide_window=Config.slide_window)

        total_error_list = train_model(network=autoencoder, dataset=dataset, n_epochs=Config.n_epochs, threshold=500)

        epochs_list.append(len(total_error_list))

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    sns.lineplot(x=epochs_list, y=lr_list, ax=ax)
    ax.set_title(f'Epoch vs LR. error threshold=500. Max number of epochs=100.'
                 f' Slide window=(128, 128), hidden layer=64')
    plt.xticks(text=f'Learning rate')
    plt.yticks(text=f'Epoch')
    plt.show()


if __name__ == '__main__':
    res_image, true_image, total_error, compression_rate = perform_pipeline()

    print(f'Current compression rate of model: {compression_rate:.3f}')

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    axs = axs.flatten()
    plt.suptitle(f'Slide window: {Config.slide_window}. Number of hidden layers: {Config.num_of_hidden_layers}')

    axs[0].imshow(res_image)
    axs[0].set_title(f'Result image')

    axs[1].imshow(true_image)
    axs[1].set_title(f'True image')

    plt.show()

    # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    #
    # sns.lineplot(x=list(range(len(total_error))), y=total_error, ax=ax)
    # ax.set_title(f'Error plots')
    # ax.set_xticklabels(f'Epoch')
    # ax.set_yticklabels(f'Error')
    #
    # plt.show()

    # iterations_depending_on_learning()