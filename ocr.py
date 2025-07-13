import json
import math
import random

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        # Initialize weights with small random values
        self.wih = [[random.uniform(-0.5, 0.5) for _ in range(self.inodes)] for _ in range(self.hnodes)]
        self.who = [[random.uniform(-0.5, 0.5) for _ in range(self.hnodes)] for _ in range(self.onodes)]

        self.learning_rate = 0.1

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def dsigmoid(self, y):
        return y * (1 - y)

    def feedforward(self, inputs):
        hidden_inputs = [sum(i*w for i, w in zip(inputs, node)) for node in self.wih]
        hidden_outputs = [self.sigmoid(x) for x in hidden_inputs]

        final_inputs = [sum(h*w for h, w in zip(hidden_outputs, node)) for node in self.who]
        final_outputs = [self.sigmoid(x) for x in final_inputs]

        return hidden_outputs, final_outputs

    def train(self, inputs, targets):
        hidden_outputs, final_outputs = self.feedforward(inputs)

        output_errors = [t - o for t, o in zip(targets, final_outputs)]
        output_gradients = [self.dsigmoid(o) * e for o, e in zip(final_outputs, output_errors)]

        for i, grad in enumerate(output_gradients):
            for j, h in enumerate(hidden_outputs):
                self.who[i][j] += self.learning_rate * grad * h

        hidden_errors = [sum(self.who[k][i] * output_gradients[k] for k in range(self.onodes)) for i in range(self.hnodes)]
        hidden_gradients = [self.dsigmoid(h) * e for h, e in zip(hidden_outputs, hidden_errors)]

        for i, grad in enumerate(hidden_gradients):
            for j, inp in enumerate(inputs):
                self.wih[i][j] += self.learning_rate * grad * inp

    def predict(self, inputs):
        _, final_outputs = self.feedforward(inputs)
        return final_outputs

    def save_weights(self, filename="data/weights.json"):
        with open(filename, 'w') as f:
            json.dump({'wih': self.wih, 'who': self.who}, f)

    def load_weights(self, filename="data/weights.json"):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.wih = data['wih']
                self.who = data['who']
        except FileNotFoundError:
            pass