"""Module with dataset for autoencoder"""

import math
from typing import Tuple

from modules.data.base_dataset import BaseDataset

import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2


class ImageAutoencoderDataset(BaseDataset):
    """Dataset for images autoencoding"""

    def __init__(self, image_path: str,
                 image_size: Tuple[int, int],
                 slide_window: Tuple[int, int] = (10, 10)):
        """


        :param image_path:
        :param image_size:
        :param slide_window: (first value for vertical, second for horizontal)
        """

        super().__init__()

        self.image_path = image_path
        self.image_size = image_size

        self.image = self.__get_image__()

        self.slide_window = slide_window

    def __get_image__(self):
        image = cv2.imread(self.image_path)

        return image

    def __get_image_chunk_iter__(self, image: np.ndarray):
        y_window, x_window = self.slide_window

        for i in range(0, image.shape[0], y_window):
            for j in range(0, image.shape[1], x_window):
                image_chunk = image[i: i + y_window, j: j + x_window]

                yield image_chunk

    def __get_image_chunk__(self, x_idx, y_idx):
        y_window, x_window = self.slide_window

        max_x = math.ceil(image.shape[1] / x_window)
        max_y = math.ceil(image.shape[0] / y_window)

        assert x_idx < max_x, 'X_idx is more than max X'
        assert x_idx >= 0, 'X_idx is less than zero'

        print(f'X_idx: {x_idx}')
        print(f'Y_idx: {y_idx}')

        assert y_idx < max_y, 'Y_idx is more than max Y'
        assert y_idx >= 0, 'Y_idx is less than zero'

        x_idx = x_idx * x_window
        y_idx = y_idx * y_window

        image_chunk = image[y_idx: y_idx + y_window, x_idx: x_idx + x_window]

        return image_chunk

    def __getitem__(self, idx):
        y_window, x_window = self.slide_window

        length_x = math.ceil(image.shape[1] / x_window)
        length_y = math.ceil(image.shape[0] / y_window)

        print(f'Length x: {length_x}')
        print(f'Length y: {length_y}')

        y_idx = idx // length_x
        x_idx = idx % length_x

        image_chunk = self.__get_image_chunk__(x_idx=x_idx, y_idx=y_idx)

        return image_chunk

    def __len__(self):
        y_window, x_window = self.slide_window

        length_x = math.ceil(image.shape[1] / x_window)
        length_y = math.ceil(image.shape[0] / y_window)

        length = length_x * length_y

        return length


if __name__ == '__main__':
    image_autoencoder_dataset = ImageAutoencoderDataset(image_path='',
                                                        image_size=(256, 256),
                                                        slide_window=(300, 300))

    image = cv2.imread('/home/vadbeg/Projects/University/MP3'
                       '/lab1/autoencoder/photo_2020-10-27_20-12-26.jpg')

    print(f'Image shape: {image.shape}')

    # for chunk in image_autoencoder_dataset.__get_image_chunk_iter__(image=image):
    #     cv2.imshow('Image', chunk)
    #     cv2.waitKey(0)

    for chunk in image_autoencoder_dataset:
        cv2.imshow(f'Image', chunk)
        cv2.waitKey(0)
