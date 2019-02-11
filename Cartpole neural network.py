"""Neural network for cartpole simulation"""
class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes  # the list sizes contains the number of neurons in the respective layers
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # shifts the output
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]  # Says how much each neuron contributes to the output

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights): # bakar ihop biases och weights för varje element i matriserna
            a = sigmoid(np.dot(w, a) + b)
        return a