"""Module with config"""


class Config:

    image_path = 'test_images/test1.jpg'

    image_size = (256, 256)
    slide_window = (32, 32)

    num_of_hidden_layers = 64
    learning_rate = 0.0003
    adaptive_lr = False

    min_error: int = 500

    n_epochs = 150


