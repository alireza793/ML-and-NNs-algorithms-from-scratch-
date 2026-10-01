import numpy as np


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class NeuralNetwork:
    def __init__(self, layer_sizes, activation="relu"):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.weights = []
        self.biases = []

        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.5
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        current = X
        for i in range(len(self.weights) - 1):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            current = relu(z)
            self.activations.append(current)

        z_out = np.dot(current, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_out)
        output = sigmoid(z_out)
        self.activations.append(output)

        return output


if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    nn = NeuralNetwork(layer_sizes=[2, 4, 4, 1])
    output = nn.forward(X)

    print("Input (XOR):")
    print(X)
    print("\nOutput (untrained network):")
    print(output)
    print("\nLayer shapes:")
    for i, (w, b) in enumerate(zip(nn.weights, nn.biases)):
        print(f"  Layer {i + 1}: W{w.shape}, b{b.shape}")

