import numpy as np
import matplotlib.pyplot as plt 
import math 

def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

def der_loss(output_layer,output_expected):
    val=(output_layer-output_expected)
    return np.dot(2,val)

