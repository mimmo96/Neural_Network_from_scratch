import numpy as np
import matplotlib.pyplot as plt
import math

def init_w(intervallo, dim_matrix):
    #intervallo = np.divide(np.sqrt(6), np.sqrt((nj+nj_plus)))
    w = np.zeros([dim_matrix[0], dim_matrix[1]])
    for i in range(dim_matrix[0]):
        for j in range(dim_matrix[1]):
            while abs(w[i,j]) < 0.0000001:
                w[i, j] = np.random.uniform(-intervallo, intervallo)
    return w

# funzione sigmoidale
def sigmoid(x):
    if -x > np.log(np.finfo(type(x)).max):
        return 0.0
    return np.divide(1, np.add(1, np.exp(-x)))

# derivata funzione sigmoidale
def derivate_sigmoid(x):
    sig = sigmoid(x)
    return np.dot(sig, np.subtract(1, sig))

def derivate_sigmoid_2(x):
    sig=np.empty(np.size(x))
    i=0
    for net in x:
        sig[i] = sigmoid(net)
        i=i+1
    return sig

# derivata loss
def der_loss(output_layer, output_expected):
    val = np.subtract(output_expected,output_layer)
    val=np.dot(2,val)
    return val

def normalize_input(x):
    colonne = x.shape[1]
    x_input = x[:, 0:(colonne-2)]
    max = x_input.max()
    min = x_input.min()

    x_input = (x_input - min)/(max -min)
    x[:, 0:(colonne-2)] = x_input
    #print(x)
    return x

def MSE(output, output_expected, example_cardinality):
    mse = np.sum(np.subtract(output, output_expected)) / example_cardinality
    mse = np.power(mse,2)
    return mse