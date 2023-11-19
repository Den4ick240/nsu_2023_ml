import numpy as np
import matplotlib.pyplot as plt
from models.nn import NeuralNet, DenseLayer, SigmoidActivation
from utils.find_best_model import find_best_model
from utils.generate_data import generate_data


def true_function_1(x):
    return (1 + np.cos(x)) / 2


def true_function_2(x):
    return (5 * x**3 + x**2 + 5) / 1400


def true_function_3(x):
    return (5 + x * np.sin(x)) / 7


def get_model(num_hidden_layers, num_hidden_nodes):
    layers = [
        DenseLayer(1, num_hidden_nodes),
        SigmoidActivation(),
    ]
    for _ in range(num_hidden_layers - 1):
        layers.append(DenseLayer(num_hidden_nodes, num_hidden_nodes))
        layers.append(SigmoidActivation())
    layers.append(DenseLayer(num_hidden_nodes, 1))
    layers.append(SigmoidActivation())
    model = NeuralNet(layers, 200, True)
    return model


def get_num_hidden_layers(model):
    return (len(model.layers) - 1) // 2


def get_num_hidden_nodes(model):
    return model.layers[0].output_size


# Зададим значения параметра M
num_hidden_layers_values = [1, 2]
num_hidden_nodes_values = [1, 2, 3]

for func, func_name in [
    (
        true_function_1,
        "cos(x)",
    ),  # For func: cos(x), Best numlayers: 2, Best num hidden nodes: 1
    (
        true_function_2,
        "5x^3 + x^2 + 5",
    ),  # For func: 5x^3 + x^2 + 5, Best numlayers: 1, Best num hidden nodes: 1
    (
        true_function_3,
        "xsin(x)",
    ),  # For func: xsin(x), Best numlayers: 2, Best num hidden nodes: 3
]:
    models = np.array(
        [
            [get_model(layers, nodes) for nodes in num_hidden_nodes_values]
            for layers in num_hidden_layers_values
        ]
    ).reshape(-1)
    x, y = generate_data(func)
    model = find_best_model(models, x, y, 2)
    model.fit(x, y)
    y_pred = model.predict(x)
    best_num_hidden_layers = get_num_hidden_layers(model)
    best_num_hidden_nodes = get_num_hidden_nodes(model)

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
        f"For func: {func_name}, Best numlayers: {best_num_hidden_layers}, Best num hidden nodes: {best_num_hidden_nodes}"
    )
