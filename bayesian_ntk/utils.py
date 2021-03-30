"""Utility functions."""

import jax.numpy as np
from jax import random
import math
from collections import namedtuple

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
        parted (bool): Set True to partition training data

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
        train_xs = random.uniform(
            x_key,
            shape=(train_points, 1),
            minval=-train_xlim,
            maxval=train_xlim
        )

    target_fn = lambda x: np.sin(x)

    train_ys = target_fn(train_xs)
    train_ys += noise_scale * random.normal(y_key, (train_points, 1))
    train = Data(
        inputs = train_xs,
        targets = train_ys
    )

    test_xs = np.linspace(-test_xlim, test_xlim, test_points)
    test_xs = np.reshape(test_xs, (test_points, 1))
    test_ys = target_fn(test_xs)
    test = Data(
        inputs = test_xs,
        targets = test_ys
    )

    return train, test
