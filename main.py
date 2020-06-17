import numpy as np

# Constants
INPUT_SIZE = 3
FEATURE_COUNT = 4

# Setting random state
np.random.seed(0)

# Input Data
X = np.random.randn(INPUT_SIZE, FEATURE_COUNT)


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        # Initializing Random weights
        self.weights = np.random.rand(n_inputs, n_neurons)

        # Initialing Bias to 0
        self.biases = np.zeros((1, n_neurons))

        self.output = 0

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = LayerDense(4, 5)
layer2 = LayerDense(5, 2)
layer1.forward(X)
layer2.forward(layer1.output)

print(X)
print(layer1.output)
print(layer2.output)