import numpy as np
from math import exp


class Node:
    def __init__(self, weights: list[float], bias: float) -> None:
        self.weights: np.ndarray = np.array(weights)
        self.bias: float = bias

    def __str__(self) -> str:
        return f"w: {self.weights}, b: {self.bias}"

    def __repr__(self) -> str:
        return f"w: {self.weights}, b: {self.bias}"

    def calc(self, inputs: list[float]) -> float:
        calcled_inp: float = np.dot(inputs, self.weights) + self.bias
        return calcled_inp / (1 + exp(calcled_inp))
