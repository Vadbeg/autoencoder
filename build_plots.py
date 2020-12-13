"""Module with plots"""

import os
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
from tqdm import tqdm
from cv2 import cv2


def perform_pipeline_for_plots():
    y_length, x_length = Config.slide_window
    num_of_nodes = y_length * x_length * 3

    autoencoder = Autoencoder(lr=Config.learning_rate, momentum=0.0, adaptive_lr=Config.adaptive_lr,
                              shape=[num_of_nodes, Config.num_of_hidden_neurons, num_of_nodes])

    dataset = ImageAutoencoderDataset(image_path=Config.image_path,
                                      image_size=Config.image_size,
                                      slide_window=Config.slide_window)

    total_error_list = train_model(network=autoencoder, dataset=dataset,
                                   n_epochs=Config.n_epochs, min_error=Config.min_error,
                                   verbose=False)

    compression_rate = calculate_image_compression(num_of_input_neurons=num_of_nodes,
                                                   num_of_chunks=len(dataset),
                                                   num_of_hidden_neurons=Config.num_of_hidden_neurons)

    return total_error_list, compression_rate


def compression_rate_vs_epochs_plot():
    hidden_layers_list = [4, 8, 16, 32, 64, 128, 256]

    num_of_epochs_list = list()
    compression_rate_list = list()

    for num_of_hidden_layers in tqdm(hidden_layers_list, postfix=f'Training networks'):
        Config.num_of_hidden_layers = num_of_hidden_layers

        total_error_list, compression_rate = perform_pipeline_for_plots()

        num_of_epochs_list.append(len(total_error_list))
        compression_rate_list.append(compression_rate)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_title(f'Compression rate vs number of epochs to achieve 0.01 MSE error')

    ax.set_xlabel('Compression rate')
    ax.set_ylabel('Number of epochs')

    sns.lineplot(x=compression_rate_list, y=num_of_epochs_list, ax=ax)

    plt.show()


def image_epochs_plot():
    image_paths_list = ['test_images/test1.jpg', 'test_images/test2.jpg',
                        'test_images/test3.jpg', 'test_images/test5.jpg',
                        'test_images/test6.jpg']
    # image_paths_list = ['test_images/test1.jpg', 'test_images/test2.jpg']

    num_of_epochs_list = list()

    for curr_image_path in tqdm(image_paths_list, postfix=f'Training networks'):
        Config.image_path = curr_image_path

        total_error_list, compression_rate = perform_pipeline_for_plots()

        num_of_epochs_list.append(len(total_error_list))

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_title(f'Image vs number of epochs to achieve 0.01 MSE error')

    xticklabels_list = list(map(lambda x: x.split(os.sep)[1], image_paths_list))
    ax.set_xticklabels(xticklabels_list)

    ax.set_xlabel('Image name')
    ax.set_ylabel('Number of epochs')

    sns.lineplot(x=image_paths_list, y=num_of_epochs_list, ax=ax)

    plt.show()


def errors_epochs_plot():
    errors_list = [0.001, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.1]

    num_of_epochs_list = list()

    for curr_error in tqdm(errors_list, postfix=f'Training networks'):
        Config.min_error = curr_error

        total_error_list, compression_rate = perform_pipeline_for_plots()

        num_of_epochs_list.append(len(total_error_list))

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_title(f'Error vs number of epochs to achieve this MSE error')

    ax.set_xlabel('Error')
    ax.set_ylabel('Number of epochs')

    sns.lineplot(x=errors_list, y=num_of_epochs_list, ax=ax)

    plt.show()


def epochs_error_for_one_training_plot():
    total_error_list, compression_rate = perform_pipeline_for_plots()

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_title(f'Epoch vs Error for one image')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')

    sns.lineplot(x=range(len(total_error_list)), y=total_error_list, ax=ax)

    plt.show()


def learning_rate_epochs_plot():
    learning_rate_list = [0.0001, 0.0003, 0.001, 0.003, 0.01]

    num_of_epochs_list = list()

    for curr_learning_rate in tqdm(learning_rate_list, postfix=f'Training networks'):
        Config.learning_rate = curr_learning_rate

        total_error_list, compression_rate = perform_pipeline_for_plots()

        num_of_epochs_list.append(len(total_error_list))

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_title(f'Learning rate vs number of epochs to achieve 0.01 MSE error')

    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Number of epochs')

    sns.lineplot(x=learning_rate_list, y=num_of_epochs_list, ax=ax)

    plt.show()


if __name__ == '__main__':
    # compression_rate_vs_epochs_plot()
    # image_epochs_plot()
    # errors_epochs_plot()
    epochs_error_for_one_training_plot()
    # learning_rate_epochs_plot()
