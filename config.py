"""Module with config"""


class Config:

    image_path = 'test_images/test3.jpg'

    image_size = (256, 256)
    slide_window = (128, 128)

    num_of_hidden_layers = 64
    learning_rate = 0.01
    adaptive_lr = False

    min_error: int = 500

    n_epochs = 200


