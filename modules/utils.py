"""Module with utils for project"""


def calculate_image_compression(num_of_input_layers: int, num_of_chunks: int, num_of_hidden_layers: int):
    """
    Calculates image compression

    :param num_of_input_layers: number of input layers in network
    :param num_of_chunks: number of chunks
    :param num_of_hidden_layers: number of hidden layers
    :return: image compression coefficient
    """

    image_compression = (num_of_input_layers * num_of_chunks) / \
                        ((num_of_input_layers + num_of_chunks) * num_of_hidden_layers + 2)
    # image_compression = 1 / image_compression

    return image_compression


