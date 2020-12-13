"""Module with config"""


class Config:

    image_path = 'test_images/test7.png'

    image_size = (256, 256)
    slide_window = (8, 8)

    num_of_hidden_neurons = 64
    learning_rate = 0.0001
    adaptive_lr = False

    min_error: float = 0.03

    n_epochs = 150


image_size = (256, 256)  # размер изображения
slide_window = (16, 16)  # размер прямоугольника для разбиения изображения

num_of_hidden_neurons = 64  # заданное число нейронов второго слоя
learning_rate = 0.001  # коэффициент обучения
min_error: float = 0.03  # допустимая ошибка

n_epochs = 150  # число итераций
