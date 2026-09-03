import math
import random

def generate_normal_distribution(N):
    rands = []
    for _ in range(N):
        u1 = random.uniform(0, 1)
        u2 = random.uniform(0, 1)
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        rands.append(z0)
    return rands

N = 10000
normal_numbers = generate_normal_distribution(N)

class Connection:
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weight = random.choice(normal_numbers)
        self.dWeight = 0.0

class Neuron:
    eta = 0.05 #learning rate
    alpha = 0.9 #momentum coefficient

    def __init__(self, layer):
        self.dendrons = []
        self.error = 0.0
        self.gradient = 0.0
        self.output = 0.0
        if layer is not None:
            for neuron in layer:
                con = Connection(neuron)
                self.dendrons.append(con)
    def dRelu(self, x):
        return 1 if x > 0 else 0

    def addError(self, err):
        self.error += err

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def dSigmoid(self, x):
        return x * (1.0 - x)

    def setError(self, err):
        self.error = err

    def setOutput(self, output):
        self.output = output

    def getOutput(self):
        return self.output

    def feedForword(self):
        if not self.dendrons:
            return
        sumOutput = sum(dendron.connectedNeuron.getOutput() * dendron.weight for dendron in self.dendrons)
        self.output = self.sigmoid(sumOutput)

    def backPropagate(self):
        self.gradient = self.error * self.dSigmoid(self.output)
        for dendron in self.dendrons:
            dendron.dWeight = Neuron.eta * (dendron.connectedNeuron.output * self.gradient) + self.alpha * dendron.dWeight
            dendron.weight += dendron.dWeight
            dendron.connectedNeuron.addError(dendron.weight * self.gradient)
        self.error = 0

class Network:
    def __init__(self, topology):
        self.layers = []
        for i, numNeuron in enumerate(topology):
            layer = []
            for _ in range(numNeuron):
                if i == 0:
                    layer.append(Neuron(None))
                else:
                    layer.append(Neuron(self.layers[-1]))
            layer.append(Neuron(None))
            layer[-1].setOutput(1)
            self.layers.append(layer)

    def setInput(self, inputs):
        for i, value in enumerate(inputs):
            self.layers[0][i].setOutput(value)

    def feedForword(self):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.feedForword()

    def backPropagate(self, target):
        for i, t in enumerate(target):
            self.layers[-1][i].setError(t - self.layers[-1][i].getOutput())
        for layer in reversed(self.layers):
            for neuron in layer:
                neuron.backPropagate()

    def getError(self, target):
        errors = [(t - n.getOutput()) ** 2 for t, n in zip(target, self.layers[-1][:-1])]
        return math.sqrt(sum(errors) / len(target))

    def getResults(self):
        return [neuron.getOutput() for neuron in self.layers[-1][:-1]]

    def getThResults(self):
        return [1 if neuron.getOutput() > 0.5 else 0 for neuron in self.layers[-1][:-1]]


def main():
    topology = [2, 5, 5, 2]
    net = Network(topology)

    inputs = [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [2, 0], [2, 2], [2, 1], [1, 2]]
    outputs = [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [2, 0], [2, 2], [2, 1], [1, 2]]

    max_epochs = 100000
    target_error = 0.01

    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(inputs)):
            net.setInput(inputs[i])
            net.feedForword()
            net.backPropagate(outputs[i])
            total_error += net.getError(outputs[i])

        avg_error = total_error / len(inputs)
        if epoch % 10000 == 0:
            print(f"Epoch {epoch}, Average Error: {avg_error:.6f}")

        if avg_error < target_error:
            print(f"Target error reached at epoch {epoch}")
            break
    else:
        print(f"Training stopped after {max_epochs} epochs")

    print("Training completed. Testing the network:")
    for i in range(len(inputs)):
        net.setInput(inputs[i])
        net.feedForword()
        result = net.getThResults()
        print(f"Input: {inputs[i]}, Expected Output: {outputs[i]}, Actual Output: {result}")

    while True:
        try:
            a = input("Type 1st input (or 'q' to quit): ")
            if a.lower() == 'q':
                break
            a = float(a)
            b = float(input("Type 2nd input: "))
            net.setInput([a, b])
            net.feedForword()
            print(f"Output: {net.getThResults()}")
        except ValueError:
            print("Please enter valid numbers.")
        except KeyboardInterrupt:
            print("\nExiting the program.")
            break


if __name__ == "__main__":
    main()
