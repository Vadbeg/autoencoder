# Image dimensionality reduction
Project with system which can reduce image dimensionality using 
linear neural network.

It uses `numpy` for matrix multiplication. Gradient descent algorithm, forward propagation and dataset loader 
are created without usage of Deep Learning algorithms. Supports only CPU computing. 

## Getting Started

To download project:
```
git clone https://github.com/Vadbeg/autoencoder.git
```


### Installing
To install all libraries you need, print in `autoencoder` directory: 

```
pip install -r requirements.txt
```

It will install all essential libraries

### Usage

After libraries installation you need to adjust configs. Config is located in `config.py` file. Config example:

```
class Config:

    image_path = 'test_images/test2.jpg'

    image_size = (256, 256)
    slide_window = (16, 16)

    num_of_hidden_layers = 64
    learning_rate = 0.001
    adaptive_lr = False

    min_error: int = 0.03

    n_epochs = 150
``` 
 
Now you can train network and perform tests:

* Training phase you can find in `start_trainig.py` script.
* Script with tests and plots you can find in `build_plots.py`.


## Built With

* [numpy](https://flask.palletsprojects.com/en/1.1.x/) - The math framework used.


## Authors

* **Vadim Titko** aka *Vadbeg* - 
[LinkedIn](https://www.linkedin.com/in/vadim-titko-89ab16149) | 
[GitHub](https://github.com/Vadbeg/PythonHomework/commits?author=Vadbeg)
 