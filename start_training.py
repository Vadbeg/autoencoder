"""Module for training setup"""

from modules.train import train_model
from modules.network.network import Autoencoder
from modules.data.dataset import ImageAutoencoderDataset
from config import Config

import numpy as np
from cv2 import cv2

if __name__ == '__main__':
    y_length, x_length = Config.slide_window
    num_of_nodes = y_length * x_length * 3

    autoencoder = Autoencoder(lr=0.001, momentum=0.1, shape=[num_of_nodes, 256, num_of_nodes])

    dataset = ImageAutoencoderDataset(image_path=Config.image_path,
                                      image_size=Config.image_size,
                                      slide_window=Config.slide_window)

    train_model(network=autoencoder, dataset=dataset, n_epochs=100)

    input_image_flatten = dataset[0][0]
    true_image_flatten = dataset[0][1]

    result = autoencoder.propagate_forward(x=input_image_flatten)

    result = result.reshape((*Config.slide_window, 3))
    result = np.int8(result)
    # result = np.clip(result, a_min=0, a_max=255)

    true_image = true_image_flatten
    true_image = true_image.reshape((*Config.slide_window, 3))

    print(f'Result: {result}')

    cv2.imshow('Result', result)
    cv2.imshow('Original', true_image)
    cv2.waitKey(0)
