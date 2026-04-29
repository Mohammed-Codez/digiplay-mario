import numpy as np
from math import exp
from activator import *
from typing import Callable


class Node:
    def __init__(
            self,
            input_count: int,
            weights: list[float] | None = None,
            bias: float | None = None,
            activator: Callable[[float], float] = silu
        ) -> None:
        self.input_count: int = input_count

        self.weights: np.ndarray = (
            np.array(weights)
            if weights is not None else 
            np.zeros(self.input_count)
        )

        self.bias: float = bias if bias is not None else 0.0

        self.activator = activator

    def __str__(self) -> str:
        return f"""
i: {self.input_count},
w: {self.weights},
b: {self.bias},
a: {self.activator.__name__}
"""

    def __repr__(self) -> str:
        return f"w: {self.weights}, b: {self.bias}"
    
    def generate_random_weights(self, min=-1, max=1) -> None:
        self.weights = np.random.uniform(min, max, self.input_count)

    def generate_random_bias(self, min=-1, max=1) -> None:
        self.bias = np.random.uniform(min, max)

    def generate_zeroed_weights(self) -> None:
        self.weights = np.zeros(self.input_count)
    
    def generate_zeroed_bias(self) -> None:
        self.bias = 0.0

    def calc(self, inputs: list[float]) -> float:
        return self.activator(np.dot(inputs, self.weights) + self.bias)

if __name__ == "__main__":
    node = Node(3)
    print(f'Node Data: {node}')
    node.generate_random_weights()
    print(f'Node Data after generating random weights: {node}')
    node.generate_random_bias()
    print(f'Node Data after generating random bias: {node}')
    print(f'Node Output: {node.calc([1, 2, 3])}')