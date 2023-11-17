import numpy as np


class MSELoss:
    def forward(self, y_pred, target):
        self.diff = y_pred - target
        return np.mean(self.diff**2)

    def backward(self):
        return 2 * self.diff


class DenseLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.zeros((output_size, 1))
        self.input = None
        self.output = None
        self.gradient_weights = None
        self.gradient_bias = None

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.weights, input_data) + self.bias
        return self.output

    def backward(self, gradient):
        self.gradient_weights = np.dot(gradient, self.input.T)
        self.gradient_bias = np.sum(gradient, axis=1, keepdims=True)
        return np.dot(self.weights.T, gradient)

    def update_parameters(self, learning_rate):
        self.weights -= learning_rate * self.gradient_weights
        self.bias -= learning_rate * self.gradient_bias


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
        self, layers, epochs=10, learning_rate=0.01, is_reg=False, loss=MSELoss()
    ):
        self.loss = loss
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.layers = layers
        self.is_reg = is_reg

    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self):
        gradient = self.loss.backward()
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def update_parameters(self, learning_rate):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.update_parameters(learning_rate)

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
            for i in range(X.shape[0]):
                # Forward pass
                input_data = (
                    X[i, :].reshape(-1, 1) if len(X.shape) > 1 else np.array([[X[i]]])
                )
                target = y_encoded[i, :].reshape(-1, 1)

                output = self.forward(input_data)

                loss = self.loss.forward(output, target)
                total_loss += loss

                self.backward()

                # Update parameters
                self.update_parameters(self.learning_rate)

            # Print average loss for the epoch
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
            output = self.forward(input_data)
            predictions.append(output.flatten())
        if self.is_reg:
            return np.array(predictions)
        return self.one_hot_decode(np.array(predictions))
