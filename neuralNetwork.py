import numpy as np


class NeuralNetwork:
    """SMALL FEEDFORWARD NETWORK WITH ReLU HIDDEN LAYERS AND A SOFTMAX OUTPUT"""

    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01, grad_clip=5.0):
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.layers = []

        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            is_output = i == len(sizes) - 2
            scale = np.sqrt(1.0 / fan_in) if is_output else np.sqrt(2.0 / fan_in)
            self.layers.append({
                "W": np.random.randn(fan_in, fan_out) * scale,
                "b": np.zeros((1, fan_out)),
            })

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x):
        self.cache = []
        self.activations = [x]

        for layer in self.layers[:-1]:
            z = np.dot(x, layer["W"]) + layer["b"]
            a = self.relu(z)
            self.cache.append(z)
            self.activations.append(a)
            x = a

        out_layer = self.layers[-1]
        z = np.dot(x, out_layer["W"]) + out_layer["b"]
        output = self.softmax(z)
        self.cache.append(z)
        self.activations.append(output)
        return output

    def _clip(self, grad):
        norm = np.linalg.norm(grad)
        if norm > self.grad_clip:
            grad = grad * (self.grad_clip / norm)
        return grad

    def backward(self, x, action_onehot, advantage):
        """POLICY-GRADIENT UPDATE: CROSS-ENTROPY ON THE CHOSEN ACTION,
        WEIGHTED BY THE ADVANTAGE (HOW MUCH BETTER/WORSE THAT ACTION DID THAN EXPECTED)"""
        batch_size = x.shape[0]

        d_out = self.activations[-1].copy()
        d_out -= action_onehot
        d_out *= advantage.reshape(-1, 1)
        d_out /= batch_size

        for i in range(len(self.layers) - 1, -1, -1):
            dW = self._clip(np.dot(self.activations[i].T, d_out))
            db = self._clip(np.sum(d_out, axis=0, keepdims=True))

            self.layers[i]["W"] -= self.learning_rate * dW
            self.layers[i]["b"] -= self.learning_rate * db

            if i > 0:
                d_out = np.dot(d_out, self.layers[i]["W"].T)
                d_out *= self.relu_derivative(self.cache[i - 1])

    def backward_mse(self, x, target):
        """PLAIN-REGRESSION UPDATE, USED FOR THE VALUE NETWORK"""
        self.forward(x)
        batch_size = x.shape[0]
        d_out = self.activations[-1] - target

        for i in range(len(self.layers) - 1, -1, -1):
            dW = self._clip(np.dot(self.activations[i].T, d_out) / batch_size)
            db = self._clip(np.sum(d_out, axis=0, keepdims=True) / batch_size)

            self.layers[i]["W"] -= self.learning_rate * dW
            self.layers[i]["b"] -= self.learning_rate * db

            if i > 0:
                d_out = np.dot(d_out, self.layers[i]["W"].T)
                d_out *= self.relu_derivative(self.cache[i - 1])

    def predict(self, state):
        return self.forward(state.reshape(1, -1))[0]

    def get_action(self, state):
        probs = self.predict(state)
        action = np.random.choice(len(probs), p=probs)
        return action, probs[action]
