import numpy as np
import math


def sigmoid(x):
    return np.divide(1, (1 +np.exp(-x)))


def derivate_sigmoid(x):
    sig = sigmoid(x)
    return np.dot(sig, np.subtract(1, sig))


def output_nn(struct_layers, x_input):
    for layer in struct_layers:
        layer.x = x_input
        x_input = layer.output()
    return x_input
