from neural_tangents import stax
from neural_tangents.stax import Dense
from jax import jit

def bann_model(
    W_std,
    b_std,
    first_layer_width,
    second_layer_width,
    subNN_num,
    keep_rate,
    activation,
    parameterization
):
    """Construct fully connected NN model and infinite width NTK & NNGP kernel
       function.

    Args:
        W_std (float): Weight standard deviation.
        b_std (float): Bias standard deviation.
        first_layer_width (int): First Hidden layer width.
        second_layer_width (int): Second Hidden layer width.
        subNN_num (int) : Number of sub neural networks in the architecture
        keep_rate (float): 1 - Dropout rate.
        activation (string): Activation function string, 'erf' or 'relu'.
        parameterization (string): Parameterization string, 'ntk' or 'standard'.

    Returns:
        `(init_fn, apply_fn, kernel_fn)`
    """
    act = activation_fn(activation)

    # multi-task learning
    # Computational Skeleton Block
    CSB = stax.serial(
        stax.FanOut(subNN_num),
        stax.parallel(
            stax.serial(
                stax.Dense(first_layer_width, W_std, b_std, parameterization=parameterization), act(),
                stax.Dense(second_layer_width, W_std, b_std, parameterization=parameterization), act(),
                stax.Dropout(keep_rate)
            ),
            stax.serial(
                stax.Dense(first_layer_width, W_std, b_std, parameterization=parameterization), act(),
                stax.Dense(2 * second_layer_width, W_std, b_std, parameterization=parameterization), act(),
                stax.Dropout(keep_rate)
            ),
            stax.serial(
                stax.Dense(first_layer_width, W_std, b_std, parameterization=parameterization), act(),
                stax.Dense(3 * second_layer_width, W_std, b_std, parameterization=parameterization), act(),
                stax.Dropout(keep_rate)
            ),
            stax.serial(
                stax.Dense(first_layer_width, W_std, b_std, parameterization=parameterization), act(),
                stax.Dense(4 * second_layer_width, W_std, b_std, parameterization=parameterization), act(),
                stax.Dropout(keep_rate)
            ),
            stax.serial(
                stax.Dense(first_layer_width, W_std, b_std, parameterization=parameterization), act(),
                stax.Dense(5 * second_layer_width, W_std, b_std, parameterization=parameterization), act(),
                stax.Dropout(keep_rate)
            ),
            stax.serial(
                stax.Dense(first_layer_width, W_std, b_std, parameterization=parameterization), act(),
                stax.Dense(6 * second_layer_width, W_std, b_std, parameterization=parameterization), act(),
                stax.Dropout(keep_rate)
            ),
            stax.serial(
                stax.Dense(first_layer_width, W_std, b_std, parameterization=parameterization), act(),
                stax.Dense(7 * second_layer_width, W_std, b_std, parameterization=parameterization), act(),
                stax.Dropout(keep_rate)
            ),
            stax.serial(
                stax.Dense(first_layer_width, W_std, b_std, parameterization=parameterization), act(),
                stax.Dense(8 * second_layer_width, W_std, b_std, parameterization=parameterization), act(),
                stax.Dropout(keep_rate)
            ),
            stax.serial(
                stax.Dense(first_layer_width, W_std, b_std, parameterization=parameterization), act(),
                stax.Dense(9 * second_layer_width, W_std, b_std, parameterization=parameterization), act(),
                stax.Dropout(keep_rate)
            ),
            stax.serial(
                stax.Dense(first_layer_width, W_std, b_std, parameterization=parameterization), act(),
                stax.Dense(10 * second_layer_width, W_std, b_std, parameterization=parameterization), act(),
                stax.Dropout(keep_rate)
            )
        ),
        stax.FanInConcat()
    )

    init_fn, apply_fn, kernel_fn = stax.serial(
        CSB,
        stax.Dropout(keep_rate),
        stax.Dense(1, W_std, b_std, parameterization=parameterization)
    )

    apply_fn = jit(apply_fn)

    return init_fn, apply_fn, kernel_fn

def homoscedastic_model(
    W_std,
    b_std,
    width,
    depth,
    activation,
    parameterization
):
    """Construct fully connected NN model and infinite width NTK & NNGP kernel
       function.

    Args:
        W_std (float): Weight standard deviation.
        b_std (float): Bias standard deviation.
        width (int): Hidden layer width.
        depth (int): Number of hidden layers.
        activation (string): Activation function string, 'erf' or 'relu'.
        parameterization (string): Parameterization string, 'ntk' or 'standard'.

    Returns:
        `(init_fn, apply_fn, kernel_fn)`
    """
    act = activation_fn(activation)

    layers_list = [Dense(width, W_std, b_std, parameterization=parameterization)]

    def layer_block():
        return stax.serial(act(), Dense(width, W_std, b_std, parameterization=parameterization))

    for _ in range(depth - 1):
        layers_list += [layer_block()]

    layers_list += [act(), Dense(1, W_std, b_std, parameterization=parameterization)]

    init_fn, apply_fn, kernel_fn = stax.serial(*layers_list)

    apply_fn = jit(apply_fn)

    return init_fn, apply_fn, kernel_fn

def activation_fn(act):
    if act == 'erf':
        return stax.Erf
    elif act == 'relu':
        return stax.Relu
