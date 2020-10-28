"""Module for training setup"""

import math
from typing import Tuple

from modules.train import train_model
from modules.network.network import Autoencoder
from modules.data.dataset import ImageAutoencoderDataset
from config import Config

import numpy as np
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

    autoencoder = Autoencoder(lr=Config.learning_rate, momentum=0.1,
                              shape=[num_of_nodes, Config.num_of_hidden_layers, num_of_nodes])

    dataset = ImageAutoencoderDataset(image_path=Config.image_path,
                                      image_size=Config.image_size,
                                      slide_window=Config.slide_window)

    train_model(network=autoencoder, dataset=dataset, n_epochs=Config.n_epochs)

    res_image = np.zeros((*Config.image_size, 3))
    true_image = dataset.image

    for idx, (input_image_flatten, true_image_flatten) in enumerate(dataset):
        result = autoencoder.propagate_forward(x=input_image_flatten)

        result = result.reshape((*Config.slide_window, 3))
        result = np.uint8(result)

        res_image = input_part_into_res_image(res_image=res_image, image_chunk=result,
                                              idx=idx, slide_window=Config.slide_window)

    res_image = np.uint8(res_image)

    return res_image, true_image


if __name__ == '__main__':
    res_image, true_image = perform_pipeline()

    print(f'Result: {res_image}')

    print(f'Result max value: {res_image.max()}')
    print(f'Result min value: {res_image.min()}')
    print(f'Result avg value: {np.average(res_image)}')

    cv2.imshow('Result', res_image)
    cv2.imshow('Original', true_image)
    cv2.waitKey(0)
