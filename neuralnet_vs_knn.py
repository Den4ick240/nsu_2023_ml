from utils.get_mnist import get_mnist
from models.nn import DenseLayer, NeuralNet, SigmoidActivation
from utils.test_model import test_model
from models.knn import KNN

mnist = get_mnist()
x_train, y_train, x_test, y_test = mnist

input_size = x_train[0].size
hidden_size = 100
output_size = y_train.max() + 1

nn_model = NeuralNet(
    [
        DenseLayer(input_size, hidden_size),
        SigmoidActivation(),
        DenseLayer(hidden_size, output_size),
        SigmoidActivation(),
    ],
    epochs=500,
)

knn_model = KNN(3)

print(f"NN accuracy: {test_model(nn_model, *mnist)}")
print(f"KNN accuracy: {test_model(knn_model, *mnist)}")
# NN accuracy: 0.9710467706013363
# KNN accuracy: 0.977728285077951
