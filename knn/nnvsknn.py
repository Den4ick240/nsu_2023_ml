from get_mnist import get_mnist
from nn import DenseLayer, NeuralNet, SigmoidActivation
from test_model import test_model
from knn import KNN

mnist = get_mnist()
x_train, y_train, x_test, y_test = mnist

input_size = x_train[0].size
hidden_size = 150
output_size = y_train.max() + 1

nn_model = NeuralNet(
    [
        DenseLayer(input_size, hidden_size),
        SigmoidActivation(),
        DenseLayer(hidden_size, output_size),
        SigmoidActivation(),
    ],
    epochs=500,
    learning_rate=0.01,
)

knn_model = KNN(3)

print(f"NN accuracy: {test_model(nn_model, *mnist)}")
print(f"KNN accuracy: {test_model(knn_model, *mnist)}")
