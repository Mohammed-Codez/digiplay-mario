import numpy as np
from random import uniform, randint
from layer import Layer
import pprint as pp
from activator import *
from typing import Callable


class NeuralNet:
    def __init__(
            self,
            input_count: int,
            layer_sizes: list[int],
            weights: list[list[list[float]]] | None = None,
            biases: list[list[float]] | None = None
        ) -> None:
        self.input_count: int = input_count
        self.layer_sizes: np.ndarray = np.array(layer_sizes)

        '''
        [
           -- represent weights per node -->
        | [[0.0 0.0]  |
        |  [0.0 0.0]] V represent nodes per layer
        | 
        | [[0.0 0.0]
        |  [0.0 0.0]]
        V
        represent layers per neural net
        ]
        '''

        self.layers: list[Layer] = [Layer(
            input_count if i == 0 else layer_sizes[i - 1],
            layer_sizes[i],
            weights[i] if weights is not None else None, 
            biases[i] if biases is not None else None
        ) for i in range(len(layer_sizes))]

    def __str__(self) -> str:
        return f"""
i: {self.input_count},
ls: {self.layer_sizes},
l:
{pp.pformat(self.layers, depth=4)}
"""
    
    def __repr__(self) -> str:
        return f"""
i: {self.input_count},
ls: {self.layer_sizes},
l:
{pp.pformat(self.layers, depth=4)}
"""
    
    def geneate_random_weights(self, min=-1, max=1) -> None:
        for layer in self.layers:
            layer.generate_random_weights(min, max)
    
    def generate_random_biases(self, min=-1, max=1) -> None:
        for layer in self.layers:
            layer.generate_random_biases(min, max)

    def generate_zeroed_weights(self) -> None:
        for layer in self.layers:
            layer.generate_zeroed_weights()

    def generate_zeroed_biases(self) -> None:
        for layer in self.layers:
            layer.generate_zeroed_biases()

    def calc(self, inputs: list[float]) -> list[float]:
        for layer in self.layers:
            inputs = layer.calc(inputs)
        return inputs

if __name__ == "__main__":
    nn = NeuralNet(4, [3, 2])
    print('NN Data: \n', nn)
    nn.geneate_random_weights()
    print('NN Data after generating random weights: \n', nn)
    nn.generate_random_biases()
    print('NN Data after generating random biases: \n', nn)
    nn.layers[-1].activator = heaviside
    print('NN Output: \n', nn.calc([1.0, 0.5, -1.5, 2.0]))