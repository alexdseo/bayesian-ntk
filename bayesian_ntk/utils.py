"""Utility functions."""

import jax.numpy as np
from jax import random
import math
from collections import namedtuple
import tensorflow_datasets as tfds

data_dir = '/tmp/tfds'  ##mnist directory

Data = namedtuple(
    'Data',
    ['inputs', 'targets']
)

def get_toy_data(
    key,
    noise_scale,
    train_points,
    test_points,
    parted = False
):
    """Fetch train and test data for Figure 1 of NeurIPS submission.
       Adds noise to train targets as per Lemma 3 of
       https://arxiv.org/abs/1806.03335

    Args:
        key: jax.random.PRNGKey instance
        noise_scale (float): output noise standard deviation
        train_points (int): Training set size
        test_points (int): Test set size
        parted (bool): Set Flase, not partitioning the training data

    Returns:
        `(train, test)`
    """
    train_xlim = np.pi
    test_xlim = np.pi
    key, x_key, y_key = random.split(key, 3)

    if not parted:
        train_xs = random.uniform(
            x_key,
            shape = (train_points, 1),
            minval = -train_xlim,
            maxval = train_xlim
        )
    else:
        half_train_points = train_points // 2
        train_xs_left = random.uniform(
            x_key,
            shape=(half_train_points, 1),
            minval=-train_xlim,
            maxval=-train_xlim / 3
        )

        train_xs_right = random.uniform(
            x_key,
            shape=(half_train_points, 1),
            minval=train_xlim / 3,
            maxval=train_xlim
        )

        train_xs = np.concatenate((train_xs_left, train_xs_right))

    target_fn = lambda x: np.sin(x)

    train_ys = target_fn(train_xs)
    train_ys += noise_scale * random.normal(y_key, (train_points, 1))
    train = Data(
        inputs=train_xs,
        targets=train_ys
    )

    test_xs = np.linspace(-test_xlim, test_xlim, test_points)
    test_xs = np.reshape(test_xs, (test_points, 1))
    test_ys = target_fn(test_xs)
    test = Data(
        inputs=test_xs,
        targets=test_ys
    )

    return train, test

def get_mnist_data(
    key,
    train_points,
    test_points
):
    """Fetch train and test data using TensorFlow data loader for larger scale experiment,
    specifically, classification task using mnist data

    Args:
        key: jax.random.PRNGKey instance
        train_points (int): Training set size
        test_points (int): Test set size

    Returns:
        `(train, test)`
    """

    def one_hot(x, k, dtype=np.float32):
        """Create a one-hot encoding of x of size k."""
        return np.array(x[:, None] == np.arange(k), dtype)

    # Fetch full datasets for evaluation
    # tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
    # You can convert them to NumPy arrays (or iterables of NumPy arrays) with tfds.dataset_as_numpy
    mnist_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=data_dir, with_info=True)
    mnist_data = tfds.as_numpy(mnist_data)
    train_data, test_data = mnist_data['train'], mnist_data['test']
    num_labels = info.features['label'].num_classes
    h, w, c = info.features['image'].shape
    num_pixels = h * w * c

    # Full train set
    train_images, train_labels = train_data['image'], train_data['label']
    train_images = np.reshape(train_images, (len(train_images), num_pixels))
    train_labels = one_hot(train_labels, num_labels)

    # Full test set
    test_images, test_labels = test_data['image'], test_data['label']
    test_images = np.reshape(test_images, (len(test_images), num_pixels))
    test_labels = one_hot(test_labels, num_labels)

    train = Data(
        inputs=train_images,
        targets=train_labels
    )

    test = Data(
        inputs=test_images,
        targets=test_labels
    )

    return train, test
