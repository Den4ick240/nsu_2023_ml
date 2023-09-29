import numpy as np
from sklearn.datasets import load_digits


def add_noise(image, noise_level=0.2):
    noisy_image = image.copy()
    mask = np.random.rand(*image.shape) < noise_level
    noise = np.random.randint(0, 16, size=image.shape, dtype=np.uint8)
    noisy_image[mask] = noise[mask]
    return noisy_image


def get_mnist(noise_level=0):
    np.random.seed(1010200)

    mnist = load_digits()
    x = mnist.data
    y = mnist.target
    test_size = 0.25
    num_samples = len(x)
    num_test_samples = int(test_size * num_samples)
    test_indicies = np.random.choice(num_samples, num_test_samples, replace=False)
    train_indicies = np.logical_not(np.isin(np.arange(num_samples), test_indicies))

    x_train = x[train_indicies]
    y_train = y[train_indicies]
    x_test = x[test_indicies]
    y_test = y[test_indicies]

    if noise_level == 0:
        noisy_x_train = x_train
    else:
        noisy_x_train = np.array([add_noise(image, noise_level) for image in x_train])
    return noisy_x_train, y_train, x_test, y_test
