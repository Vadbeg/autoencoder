"""Module with dataset for autoencoder"""

import math
from typing import Tuple, Generator

from modules.data.base_dataset import BaseDataset

import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2


class ImageAutoencoderDataset(BaseDataset):
    """Dataset for images autoencoding"""

    def __init__(self, image_path: str,
                 image_size: Tuple[int, int] = None,
                 slide_window: Tuple[int, int] = (10, 10)):
        """
        Dataset for image autoencoder netowork

        :param image_path: path to the image
        :param image_size: size of the image
        :param slide_window: slided window which will iterate over image
        """

        super().__init__()

        self.image_path = image_path
        self.image_size = image_size

        self.image = self.__get_image__()

        self.slide_window = slide_window

    def __get_image__(self) -> np.ndarray:
        """
        Reads image from give path. If self.image_size is not None, resizes image.

        :return: image
        """

        image = cv2.imread(self.image_path)

        if self.image_size:
            image = cv2.resize(image, dsize=self.image_size)

        return image

    @staticmethod
    def __normalize_image__(image) -> np.ndarray:
        """
        Performs image normalization

        :param image: original image
        :return: normalized image
        """

        image_norm = image / 255

        return image_norm

    def __get_image_chunk_iter__(self, image: np.ndarray) -> Generator[np.ndarray]:
        """
        Iterator for getting image chunks depending on self.slide_window

        :param image: original image
        :return: image chunks generator
        """

        y_window, x_window = self.slide_window

        for i in range(0, image.shape[0], y_window):
            for j in range(0, image.shape[1], x_window):
                image_chunk = image[i: i + y_window, j: j + x_window]

                yield image_chunk

    def __get_image_chunk__(self, x_idx: int, y_idx: int) -> np.ndarray:
        """
        Reads image chunk by idx

        :param x_idx: vertical index of chunk
        :param y_idx: horizontal index of chunk
        :return: image chunk by idx
        """

        y_window, x_window = self.slide_window

        max_x = math.ceil(self.image.shape[1] / x_window)
        max_y = math.ceil(self.image.shape[0] / y_window)

        assert x_idx < max_x, 'X_idx is more than max X'
        assert x_idx >= 0, 'X_idx is less than zero'

        assert y_idx < max_y, 'Y_idx is more than max Y'
        assert y_idx >= 0, 'Y_idx is less than zero'

        x_idx = x_idx * x_window
        y_idx = y_idx * y_window

        if y_idx + y_window > self.image.shape[0]:
            y_idx = self.image.shape[0] - y_window

        if x_idx + x_window > self.image.shape[1]:
            x_idx = self.image.shape[1] - x_window

        image_chunk = self.image[y_idx: y_idx + y_window, x_idx: x_idx + x_window]

        # It is not necessary, but still it makes me feel better
        if image_chunk.shape[:2] != self.slide_window:
            image_chunk = cv2.resize(image_chunk, dsize=self.slide_window)

        return image_chunk

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets image chunks by idx from image.

        idx = y_idx * length_x + x_idx

        :param idx: index of chunk
        :return: chunk
        """

        if idx >= self.__len__():
            raise StopIteration

        y_window, x_window = self.slide_window

        length_x = math.ceil(self.image.shape[1] / x_window)
        length_y = math.ceil(self.image.shape[0] / y_window)

        y_idx = idx // length_x
        x_idx = idx % length_x

        image_chunk = self.__get_image_chunk__(x_idx=x_idx, y_idx=y_idx)
        image_chunk_norm = self.__normalize_image__(image_chunk)

        image_flatten = image_chunk.flatten()
        image_flatten_norm = image_chunk_norm.flatten()

        return image_flatten_norm, image_flatten_norm
        # return image_chunk

    def __len__(self) -> int:
        """
        Calculates length of dataset

        :return: length of dataset
        """

        y_window, x_window = self.slide_window

        length_x = math.ceil(self.image.shape[1] / x_window)
        length_y = math.ceil(self.image.shape[0] / y_window)

        length = length_x * length_y

        return length


if __name__ == '__main__':
    image_path = '/home/vadbeg/Projects/University/MP3/lab1/autoencoder/test_images/test1.jpg'

    image_autoencoder_dataset = ImageAutoencoderDataset(image_path=image_path,
                                                        image_size=(256, 256),
                                                        slide_window=(200, 200))

    print(image_autoencoder_dataset[0])

    for idx, chunk in enumerate(image_autoencoder_dataset):
        chunk = chunk[1]

        chunk = np.resize(chunk, new_shape=(200, 200, 3))

        cv2.imshow(f'Image', chunk)

        print(f'Chunk size: {chunk.shape}')

        cv2.waitKey(0)
