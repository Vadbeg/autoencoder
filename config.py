"""Module with config"""


class Config:

    image_path = 'test_images/test4.webp'

    image_size = (256, 256)
    slide_window = (16, 16)

    num_of_hidden_layers = 32
    learning_rate = 0.003
    adaptive_lr = False

    min_error: int = 500

    n_epochs = 150


