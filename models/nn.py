import numpy as np
from numpy._typing import ArrayLike


class MSELoss:
    def forward(self, y_pred, target):
        self.diff = y_pred - target
        return np.mean(self.diff**2)

    def backward(self):
        return 2 * self.diff


class DenseLayer:
    def __init__(self, size):
        self.output_size = size
        self.input_size = 0
        self.weights = np.array([])
        self.bias = np.array([])
        self.input = None
        self.output = None
        self.gradient_weights = None
        self.gradient_bias = None

    def init_weights(self, input_size):
        output_size = self.output_size
        self.input_size = input_size
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.zeros((output_size, 1))

    def forward(self, input_data):
        if self.input_size == 0:
            self.init_weights(input_data.shape[1])
        self.input = input_data
        self.output = np.array(
            [
                np.dot(self.weights, input_column) + self.bias
                for input_column in input_data
            ]
        )
        return self.output

    def backward(self, gradient):
        self.gradient_weights = np.sum(
            [
                np.dot(_gradient, _input.T)
                for _gradient, _input in zip(gradient, self.input)
            ],
            axis=0,
        )
        self.gradient_bias = np.sum(gradient.sum(axis=0), axis=1, keepdims=True)
        return np.array([np.dot(self.weights.T, _gradient) for _gradient in gradient])


class Batchnorm:
    def __init__(self):
        self.gamma = None
        self.beta = None
        self.global_mu_var = None
        self.batch_count = 0

    def init_params(self):
        self.gamma = np.ones((1, self.X_shape[1]))
        self.beta = np.zeros((1, self.X_shape[1]))

    def forward(self, X):
        self.n_X = X.shape[0]
        self.X_shape = X.shape
        if self.gamma is None:
            self.init_params()

        self.X_flat = X.ravel().reshape(self.n_X, -1)
        if self.n_X != 1:
            self.mu = np.mean(self.X_flat, axis=0)
            self.var = np.var(self.X_flat, axis=0)
        else:
            self.mu, self.var = self.global_mu_var

        self.X_norm = (self.X_flat - self.mu) / np.sqrt(self.var + 1e-8)
        out = np.array([self.gamma * it + self.beta for it in self.X_norm])

        return out.reshape(self.X_shape)

    def backward(self, dout):
        dout = dout.ravel().reshape(dout.shape[0], -1)
        X_mu = self.X_flat - self.mu
        var_inv = 1.0 / np.sqrt(self.var + 1e-8)

        self.dbeta = np.sum(dout, axis=0)
        self.dgamma = np.sum(dout * self.X_norm, axis=0)

        dX_norm = dout * self.gamma
        dvar = np.sum(dX_norm * X_mu, axis=0) * -0.5 * (self.var + 1e-8) ** (-3 / 2)
        dmu = np.sum(dX_norm * -var_inv, axis=0) + dvar * 1 / self.n_X * np.sum(
            -2.0 * X_mu, axis=0
        )
        dX = (dX_norm * var_inv) + (dmu / self.n_X) + (dvar * 2 / self.n_X * X_mu)

        dX = dX.reshape(self.X_shape)
        if self.batch_count == 0:
            self.global_mu_var = np.array([self.mu, self.var])
            self.batch_count += 1
        else:
            self.global_mu_var *= self.batch_count
            self.batch_count += 1
            self.global_mu_var = (
                self.global_mu_var + np.array([self.mu, self.var])
            ) / self.batch_count

        return dX


class WeightUpdater:
    def __init__(self, learning_rate) -> None:
        self.learning_rate = learning_rate

    def update_layer(self, layer: DenseLayer) -> None:
        layer.weights = layer.weights - self.learning_rate * layer.gradient_weights
        layer.bias = layer.bias - self.learning_rate * layer.gradient_bias

    def update_batchnorm_layer(self, layer: Batchnorm) -> None:
        layer.beta -= self.learning_rate * layer.dbeta
        layer.gamma -= self.learning_rate * layer.dgamma


class SigmoidActivation:
    def __init__(self):
        self.input = None
        self.output = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, input_data):
        self.input = input_data
        self.output = self.sigmoid(input_data)
        return self.output

    def backward(self, gradient):
        return gradient * self.output * (1 - self.output)


class NeuralNet:
    def __init__(
        self,
        layers,
        epochs=10,
        is_reg=False,
        loss=MSELoss(),
        weights_updater=WeightUpdater(0.1),
        batch_size=16,
    ):
        self.loss = loss
        self.epochs = epochs
        self.layers = layers
        self.is_reg = is_reg
        self.weights_updater = weights_updater
        self.batch_size = batch_size

    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self):
        gradient = self.loss.backward()
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def update_parameters(self):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                self.weights_updater.update_layer(layer)
            elif isinstance(layer, Batchnorm):
                self.weights_updater.update_batchnorm_layer(layer)

    def one_hot_encode(self, labels, num_classes):
        encoded_labels = np.zeros((len(labels), num_classes))
        for i, label in enumerate(labels):
            encoded_labels[i, int(label)] = 1
        return encoded_labels

    def one_hot_decode(self, one_hot_array):
        return np.argmax(one_hot_array, axis=1)

    def fit(self, X, y):
        if not self.is_reg:
            num_classes = len(np.unique(y))
            y_encoded = self.one_hot_encode(y, num_classes)
        else:
            y_encoded = y.reshape(-1, 1)
        for epoch in range(self.epochs):
            total_loss = 0
            X_shuffled = X.copy()
            y_shuffled = y_encoded.copy()
            np.random.shuffle(X_shuffled)
            np.random.shuffle(y_shuffled)

            for i in range(0, X.shape[0], self.batch_size):
                # Forward pass
                batch_size = self.batch_size
                ii = min(i, X.shape[0] - batch_size)
                input_data = X[ii : ii + batch_size, :].reshape(batch_size, -1, 1)
                target = y_encoded[ii : ii + batch_size, :].reshape(batch_size, -1, 1)

                output = self.forward(input_data)

                loss = self.loss.forward(output, target)
                total_loss += loss

                self.backward()

                # Update parameters
                self.update_parameters()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / X.shape[0]}"
                )

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            input_data = (
                X[i, :].reshape(-1, 1) if len(X.shape) > 1 else np.array([[X[i]]])
            )
            output = self.forward(np.array([input_data]))
            predictions.append(output[0].flatten())
        if self.is_reg:
            return np.array(predictions)
        return self.one_hot_decode(np.array(predictions))
