# Module to import functions from in examples for multiprocessing backend
import numpy as np


def stochastic_function_seeded(max_value, random_state):
    rng = np.random.RandomState(random_state)
    return rng.randint(max_value, size=5)


def stochastic_function(max_value):
    """Randomly generate integer up to a maximum value."""
    return np.random.randint(max_value, size=5)


def func_async(i, *args):
    """Asynchronous function to multiply the first argument by two."""
    return 2 * i
