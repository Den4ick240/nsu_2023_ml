import numpy as np
import matplotlib.pyplot as plt
from nn import NeuralNet, DenseLayer, SigmoidActivation


# Зададим функции, которые мы хотим восстановить
def true_function_1(x):
    return (1 + np.cos(x)) / 2


def true_function_2(x):
    return (5 * x**3 + x**2 + 5) / 1400


def true_function_3(x):
    return (5 + x * np.sin(x)) / 7


# Зададим функцию для создания выборки с ошибками измерения
def generate_data(func, x, error_type="uniform", scale=0.1):
    if error_type == "uniform":
        error = np.random.uniform(-scale, scale, len(x))
    elif error_type == "normal":
        error = np.random.normal(0, scale, len(x))
    y = func(x) + error
    return y


def regression(x, y, num_hidden_layers, num_hidden_nodes):
    layers = [
        DenseLayer(1, num_hidden_nodes),
        SigmoidActivation(),
    ]
    for _ in range(num_hidden_layers - 1):
        layers.append(DenseLayer(num_hidden_nodes, num_hidden_nodes))
        layers.append(SigmoidActivation())
    layers.append(DenseLayer(num_hidden_nodes, 1))
    layers.append(SigmoidActivation())
    model = NeuralNet(layers, 200, 0.1, True)
    model.fit(x, y)
    return model


# Функция для оценки регрессии на новых данных
def predict(x, model):
    return model.predict(x)


# Зададим диапазон значений x
x = np.linspace(0, 2 * np.pi, 100)

# Зададим значения параметра M
num_hidden_layers_values = [1, 2]

num_hidden_nodes_values = [1, 2, 3]

for func, func_name in [
    (true_function_1, "cos(x)"),
    (true_function_2, "5x^3 + x^2 + 5"),
    (true_function_3, "xsin(x)"),
]:
    best_num_hidden_layers = -1
    best_num_hidden_nodes = -1
    best_mse = float("inf")
    for num_hidden_layers in num_hidden_layers_values:
        for num_hidden_nodes in num_hidden_nodes_values:
            mse_sum = 0

            y = generate_data(func, x, error_type="uniform", scale=0.1)
            folds = 2
            fold_size = len(y) // folds

            for i in range(folds):  # Повторяем 10 раз для разных выборок
                test_start = i * fold_size
                test_end = test_start + fold_size
                x_test = x[test_start:test_end]
                y_test = y[test_start:test_end]
                x_train = np.concatenate((x[:test_start], x[test_end:]))
                y_train = np.concatenate((y[:test_start], y[test_end:]))

                model = regression(
                    x_train, y_train, num_hidden_layers, num_hidden_nodes
                )
                y_pred = predict(x_test, model)

                mse = np.mean((y_pred - y_test) ** 2)
                mse_sum += mse

            avg_mse = mse_sum / 10

            if avg_mse < best_mse:
                best_num_hidden_layers = num_hidden_layers
                best_num_hidden_nodes = num_hidden_nodes
                best_mse = avg_mse

    y = generate_data(func, x, error_type="uniform", scale=0.1)
    model = regression(x, y, best_num_hidden_layers, best_num_hidden_nodes)
    y_pred = predict(x, model)

    plt.figure()
    plt.plot(x, y, "bo")
    plt.plot(x, func(x), label="True Function")
    plt.plot(x, y_pred, label="Polynomial Regression")
    plt.title(
        f"{func_name}, layers={best_num_hidden_layers}, nodes={best_num_hidden_nodes}"
    )
    plt.legend()
    plt.show()
    # plt.savefig(f"{func_name}_{M}.png")

    print(
        f"For func: {func_name}, Best numlayers: {best_num_hidden_layers}, Best num hidden nodes: {best_num_hidden_nodes}, Best Average MSE: {best_mse}"
    )
