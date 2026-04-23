import numpy as np
from node import Node


class Layer:
    def __init__(self, weights: list[list[float]], biases: list[float]) -> None:
        self.weights: np.ndarray = np.array(weights)
        self.biases: np.ndarray = np.array(biases)

    def __str__(self) -> str:
        return f"w: {self.weights}, b: {self.biases}"

    def __repr__(self) -> str:
        return f"w: {self.weights}, b: {self.biases}"

    def calc(self, inputs: list[float]):
        result = []
        print(inputs)
        for i in range(len(self.weights)):
            node = Node(self.weights[i], self.biases[i])
            print(node)
            output = node.calc(inputs)
            print(output)
            result.append(output)

        return result


layer = Layer(
    [
        [0.1, 0.2],  # first node
        [0.4, 0.3],  # second node
    ],
    [
        0.8,  # first node
        0.5,  # second node
    ],
)
print(layer)
print(layer.calc([0.7, 0.6]))
