import numpy as np
from models.polynomregressor import PolynomRegressor
from utils.find_best_model import find_best_model
from utils.generate_data import generate_data
import matplotlib.pyplot as plt


def true_function_1(x):
    return np.cos(x)


def true_function_2(x):
    return 5 * x**3 + x**2 + 5


def true_function_3(x):
    return x * np.sin(x)


M_values = range(16)
alphas = [0, 0.001, 0.01, 0.1, 1, 10]

for func, func_name in [
    (true_function_1, "cos(x)"),
    (true_function_2, "5x^3 + x^2 + 5"),
    (true_function_3, "xsin(x)"),
]:
    models = np.array(
        [[PolynomRegressor(M, alpha) for alpha in alphas] for M in M_values]
    ).reshape(-1)
    x, y = generate_data(func)
    model = find_best_model(models, x, y)
    model.fit(x, y)
    y_pred = model.predict(x)
    M = model.M
    alpha = model.alpha

    plt.figure()
    plt.plot(x, y, "bo")
    plt.plot(x, func(x), label="True Function")
    plt.plot(x, y_pred, label="Polynomial Regression")
    plt.title(f"{func_name}, M={M}, alpha={alpha}")
    plt.legend()
    plt.show()
    # plt.savefig(f"{func_name}_{M}.png")
