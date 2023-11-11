import numpy as np


def generate_data(func, error_type="uniform", scale=0.1):
    x = np.linspace(0, 2 * np.pi, 100)
    if error_type == "uniform":
        error = np.random.uniform(-scale, scale, len(x))
    else:
        error = np.random.normal(0, scale, len(x))
    y = func(x) + error
    return x, y
