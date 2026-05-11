import numpy as np
from node import Node
from activator import *
from typing import Callable

import node


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

        self._activator: Callable[[float], float] = activator

        '''
         -- represent weights per node -->
        [[0.0 0.0]  |
         [0.0 0.0]] V represent nodes per layer
        '''

        self._nodes: list[Node] = [Node(
            input_count,
            self.weights[i],
            self.biases[i],
            activator
        ) for i in range(node_count)]

    @property
    def nodes(self) -> list[Node]:
        return self._nodes
    

    @nodes.setter
    def nodes(self, value: list[Node]) -> None:
        self._nodes = value

        self.node_count = len(value)
        self.weights = np.array([node.weights for node in value])
        self.biases = np.array([node.bias for node in value])


    @property
    def activator(self) -> Callable[[float], float]:
        return self._activator

    @activator.setter
    def activator(self, value: Callable[[float], float]) -> None:
        self._activator = value

        for node in self.nodes:
            node.activator = value

    def __str__(self) -> str:
        return f"""
i: {self.input_count},
n: {self.node_count},
w:
{self.weights},
b:
{self.biases},
a: {self.activator.__name__},
nodes:
{self.nodes}
"""

    def __repr__(self) -> str:
        return f"w: {self.weights}, b: {self.biases}"
    
    def generate_random_weights(self, min=-1, max=1) -> None:
        self.weights = np.random.uniform(min, max, (self.node_count, self.input_count))
        for i, node in enumerate(self.nodes):
            node.weights = self.weights[i]

    def generate_random_biases(self, min=-1, max=1) -> None:
        self.biases = np.random.uniform(min, max, self.node_count)
        for i, node in enumerate(self.nodes):
            node.bias = self.biases[i]

    def generate_zeroed_weights(self) -> None:
        self.weights = np.zeros((self.node_count, self.input_count))
        for i, node in enumerate(self.nodes):
            node.weights = self.weights[i]

    def generate_zeroed_biases(self) -> None:
        self.biases = np.zeros(self.node_count)
        for i, node in enumerate(self.nodes):
            node.bias = self.biases[i]

    def calc(self, inputs: list[float], thing = 0):
        return [node.calc(inputs, thing) for node in self.nodes]


if __name__ == "__main__":
    layer = Layer(2, 3)
    print(f'Layer Data: \n{layer}')
    layer.generate_random_weights()
    print(f'Layer Data after generating random weights: \n{layer}')
    layer.generate_random_biases()
    print(f'Layer Data after generating random biases: \n{layer}')
    print(f'Layer Output: \n{layer.calc([1.0, 0.5])}')
    layer.activator = heaviside
    print(f'Layer Output after converting to Heaveside: \n{layer.calc([1.0, 0.5])}')