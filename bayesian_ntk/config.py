"""
Default hyperparameter values
"""
from functools import partial

NOISE_SCALE = 5e-1
TRAIN_NOISE_SCALE = 1e-1
ENSEMBLE_SIZE = 20

_model_configs = {
    "default": dict(
        W_std = 1.5,                # Weight standard deviation
        b_std = 0.05,               # Bias standard deviation
        width = 512,                # Hidden layer width
        depth = 2,                  # Number of hidden layers
        activation = 'erf'          # Activation function
    ),
    
    "bann": dict(
        W_std = 1.5,                # Weight standard deviation
        b_std = 0.05,               # Bias standard deviation
        first_layer_width = 2,      # First Hidden layer width
        second_layer_width = 5,     # Second Hidden layer width
        keep_rate = 0.9,            # Dropout rate
        subNN_num = 10,             # Number of sub neural networks
        activation = 'erf'          # Activation function
    ),

    "bann_mnist": dict(
        W_std = 1.5,                # Weight standard deviation
        b_std = 0.05,               # Bias standard deviation
        first_layer_width = 2,      # First Hidden layer width
        second_layer_width = 5,     # Second Hidden layer width
        keep_rate = 0.9,            # Dropout rate
        subNN_num = 10,             # Number of sub neural networks
        activation = 'relu'        # Activation function for classification task
    )
}

_train_configs = {
    "default": dict(
        learning_rate = 1e-3,       # Learning rate
        training_steps = 50000,     # Number of gradient updates
        noise_scale = NOISE_SCALE,  # Observation noise standard deviation
        **_model_configs["default"]
    ),

    "bann": dict(
        learning_rate = 1e-3,       # Learning rate
        training_steps = 50000,     # Number of gradient updates
        noise_scale = NOISE_SCALE,  # Observation noise standard deviation
        **_model_configs["bann"]
    ),

    "bann_mnist": dict(
        learning_rate = 1e-2,       # Learning rate
        epochs = 20,                # Number of gradient updates
        batch_size = 128,           # batch size
        n_targets = 10,             # Number of target labels
        layer_sizes=[784, 512, 512, 10],  # layer size of mnist image
        ** _model_configs["bann_mnist"]
    )
}

_data_configs = {
    "default": dict(
        train_points = 20,          # Training set size
        test_points = 50            # Test set size
    ),

    "mnist": dict(
        train_points = 60000,          # Training set size from tfds
        test_points = 10000            # Test set size from tfds
    )
}

def get_model_config(name, _cfg_dct=_model_configs):
    try:
        return _cfg_dct[name.lower()]
    except KeyError:
        raise ValueError(
            f"Could not find config {name} in config.py."
            f"Available configs are: {list(_cfg_dct.keys())}"
        )

get_train_config = partial(get_model_config, _cfg_dct=_train_configs)
get_data_config = partial(get_model_config, _cfg_dct=_data_configs)
