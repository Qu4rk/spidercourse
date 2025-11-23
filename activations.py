import sys
sys.path.append('.') # Tells Python to look in the root folder

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

def softmax(x):
    stableX = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(stableX)
    return exp / np.sum(exp, axis=1, keepdims=True)


def softmaxDeriv(x):
    s = softmax(x)
    return s * (1 - s)

def Linear(x):
    return x

def LinearDeriv(x):
    return np.ones_like(x)