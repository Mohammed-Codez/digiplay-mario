import numpy as np

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def silu(x: float) -> float:
    return x * sigmoid(x)

def relu(x: float) -> float:
    return np.maximum(0, x)

def tanh(x: float) -> float:
    return np.tanh(x)

def leaky_relu(x: float, alpha=0.01) -> float:
    return np.maximum(alpha * x, x)

def softmax(x: list[float]) -> list[float]:
    exp_x = np.exp(x - np.max(x))  # for numerical stability
    return exp_x / exp_x.sum()

def heaviside(x: float) -> float:
    return np.heaviside(x, 0)