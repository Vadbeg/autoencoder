"""Module with neural network code"""

from typing import List, Union

import numpy as np

from modules.network.utils import (sigmoid, sigmoid_der,
                                   mse_loss,
                                   mse_loss_der)


class Autoencoder:
    """Neural network"""

    def __init__(self, momentum: float, shape: List[int], adaptive_lr: bool = False, lr: float = 0.03):
        self.lr = lr
        self.adaptive_lr = adaptive_lr
        self.momentum = momentum

        self.shape = shape

        self.n_layers = len(shape)

        self.layers = self.__init_layers__()
        self.weights = self.__init_weights__()

        self.dw = [0] * len(self.weights)

    def __init_layers__(self) -> List[np.ndarray]:
        """
        Initialize layers of NN

        :return: list of initialized layers
        """

        layers = list()

        layers.append(np.ones(self.shape[0]))

        for i in range(1, self.n_layers):
            layers.append(np.ones(self.shape[i]))

        return layers

    def __init_weights__(self) -> List[np.ndarray]:
        """
        Neural networks he initialization

        :url: https://medium.com/@prateekvishnu/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528
        :return: list of weights
        """

        if len(self.layers) == 0:
            raise ValueError('Before weight initialization, initialize layers!')

        weights = list()

        for i in range(self.n_layers - 1):
            curr_weights = np.random.randn(self.layers[i].size,
                                           self.layers[i + 1].size) + np.sqrt(2 / self.layers[i].size)

            weights.append(curr_weights)

        return weights

    def propagate_forward(self, x: Union[np.ndarray, List]) -> np.ndarray:
        """
        Propagate data in network forward. Forward pass

        :param x: data to propagate
        :return: result of neural network
        """

        self.layers[0] = x

        if self.adaptive_lr:
            self.lr = 10 / (np.dot(x, x.T))

        for i in range(1, len(self.shape) - 1):
            self.layers[i][...] = sigmoid(
                np.dot(self.layers[i - 1], self.weights[i - 1])
            )

        if len(self.shape) - 2 >= 0:
            last_idx = len(self.shape) - 1

            self.layers[last_idx][...] = np.dot(self.layers[last_idx - 1], self.weights[last_idx - 1])

        return self.layers[-1]

    def propagate_backward(self, target) -> List[float]:
        """
        Performs backpropagation on neural network.

        :param target: desired result
        :return: error of the network
        """

        deltas = list()

        cross_entropy_loss_number = mse_loss(y_pred=self.layers[-1],
                                             y_true=target)
        last_layer_delta = mse_loss_der(y_pred=self.layers[-1],
                                        y_true=target)

        deltas.append(last_layer_delta)

        # TODO: add try catch clause here for negative len(self.shape) - 2
        for i in range(len(self.shape) - 2, 0, -1):
            curr_delta = np.dot(deltas[0],
                                self.weights[i].T * sigmoid_der(self.layers[i]))

            deltas.insert(0, curr_delta)

        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            curr_delta = np.atleast_2d(deltas[i])

            curr_dw = np.dot(layer.T, curr_delta)

            self.weights[i] += self.lr * curr_dw + self.lr * self.momentum * self.dw[i]

            self.dw[i] = curr_dw

        return cross_entropy_loss_number


if __name__ == '__main__':
    network = Autoencoder(lr=0.01, momentum=0.1, shape=[10, 3, 10])

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = network.propagate_forward(x=x)

    print(f'Result: {result}')

    error = network.propagate_backward(target=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    print(f'Result: {result}')
    print(f'Error: {error}')

