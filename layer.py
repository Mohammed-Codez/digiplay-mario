import numpy as np
from node import Node
from activator import *
from typing import Callable


class Layer:
    def __init__(
            self,
            input_count: int,
            node_count: int,
            weights: list[list[float]] | None = None,
            biases: list[float] | None = None,
            activator: Callable[[float], float] = silu
        ) -> None:
        self.input_count: int = input_count
        self.node_count: int = node_count

        self.weights: np.ndarray = (
            np.array(weights)
            if weights is not None else
            np.zeros((self.node_count, self.input_count))
        )

        self.biases: np.ndarray = (
            np.array(biases)
            if biases is not None
            else np.zeros(self.node_count)
        )

        self.activator = activator

    def __str__(self) -> str:
        return f"""
i: {self.input_count},
n: {self.node_count},
w:
{self.weights},
b:
{self.biases},
a: {self.activator.__name__}
"""

    def __repr__(self) -> str:
        return f"w: {self.weights}, b: {self.biases}"
    
    def generate_random_weights(self, min=-1, max=1) -> None:
        self.weights = np.random.uniform(min, max, (self.node_count, self.input_count))

    def generate_random_biases(self, min=-1, max=1) -> None:
        self.biases = np.random.uniform(min, max, self.node_count)

    def generate_zeroed_weights(self) -> None:
        self.weights = np.zeros((self.node_count, self.input_count))

    def generate_zeroed_biases(self) -> None:
        self.biases = np.zeros(self.node_count)

    def calc(self, inputs: list[float]):
        return [
            self.activator(np.dot(self.weights[i], inputs) + self.biases[i])
            for i in range(self.node_count)
        ]


if __name__ == "__main__":
    layer = Layer(2, 3)
    print(f'Layer Data: \n{layer}')
    layer.generate_random_weights()
    print(f'Layer Data after generating random weights: \n{layer}')
    layer.generate_random_biases()
    print(f'Layer Data after generating random biases: \n{layer}')
    print(f'Layer Output: \n{layer.calc([1.0, 0.5])}')