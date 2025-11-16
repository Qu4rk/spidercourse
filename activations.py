import numpy as np

def step(x):
    return np.array(x > 0, dtype=int)

def stepDeriv(x):
    return np.zeros_like(x)

def ReLU(x):
    return np.maximum(0, x)

def ReLUDeriv(x):
    return (x > 0).astype(np.float64)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidDeriv(x):
    s = sigmoid(x)
    return s * (1 - s)

#updated for batches and large numbers
def softmax(x):
    stableX = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(stableX)
    return exp / np.sum(exp, axis=1, keepdims=True)


def softmaxDeriv(x):
    s = softmax(x)
    return s * (1 - s)

def Linear(x):
    # No activation, just pass the value through
    return x

def LinearDeriv(x):
    # The derivative of x is 1
    # We use np.ones_like to keep the shape correct
    return np.ones_like(x)