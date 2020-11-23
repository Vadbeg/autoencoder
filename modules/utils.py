"""Module with utils for project"""


def calculate_image_compression(num_of_hidden_neurons: int, num_of_chunks: int, num_of_input_neurons: int):
    """
    Calculates image compression

    :param num_of_input_neurons: number of input layers in network
    :param num_of_chunks: number of chunks
    :param num_of_hidden_neurons: number of hidden layers
    :return: image compression coefficient
    """

    image_compression = (num_of_input_neurons * num_of_chunks) / \
                        ((num_of_input_neurons + num_of_chunks) * num_of_hidden_neurons + 2)

    return image_compression


