import numpy as np
import math


import numpy as np
import matplotlib.pyplot as plt 
import math 

#funzione sigmoidale
def sigmoid(x):
    return np.divide(1, (1 +np.exp(-x)))

#derivata funzione sigmoidale
def derivate_sigmoid(x):
    sig = sigmoid(x)
    return np.dot(sig, np.subtract(1, sig))

#derivata loss
def der_loss(output_layer,output_expected):
    val=(output_layer-output_expected)
    return np.dot(2,val)

#funzione costo layer
def dCdW(Di,Do,Dc):
    val=np.dot(Di,Do)
    val2=np.dot(val,Dc)
    return val2

#output NN
def output_nn(struct_layers, x_input):
    for layer in struct_layers:
        layer.x = x_input
        x_input = layer.output()
    return x_input

