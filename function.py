import numpy as np
import matplotlib.pyplot as plt
import math

def init_w(nj, nj_plus, dim_matrix):
    intervallo = np.divide(np.sqrt(6), np.sqrt((nj+nj_plus)))
    w = np.zeros([dim_matrix[0], dim_matrix[1]])
    for i in range(dim_matrix[0]):
        for j in range(dim_matrix[1]):
            while abs(w[i,j]) < 0.0000001:
                w[i, j] = np.random.uniform(-0.5, 0.5)
    return w

# funzione sigmoidale
def sigmoid(x):
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


# funzione costo layer
def c(Di, Do, Dc):
    return 0


def output_lastlayer(x_input,layer):
    output=np.zeros(layer.nj)
    for nj in range(layer.nj):
        output[nj]=layer.net(nj, x_input)
    return output

def output_nn(struct_layers, x_input, row_input_layer):
    i = np.size(struct_layers)-1
    for layer in struct_layers:
        x_input = np.append(x_input, 1)
        layer.x[row_input_layer,:] = x_input  
        #hidden layer
        if i != 0:
            x_input = layer.output(x_input)
        #output layer
        else:
            x_input = output_lastlayer(x_input,layer)
        i = i - 1
    return x_input

def normalize_input(x):
    colonne = x.shape[1]
    x_input = x[:, 0:(colonne-2)]
    max = x_input.max()
    min = x_input.min()

    x_input = (x_input - min)/(max -min)
    x[:, 0:(colonne-2)] = x_input
   # print(x)
    return x
    """size_x = len(x)
    for i in range(size_x):
        tmp = np.max(x[i])
        if tmp > max:
            max = tmp
        tmp = np.min(x[i])
        if tmp < min:
            min = tmp
    for i in range(size_x):
        x[i] = np.divide(np.subtract(x[i],min), np.subtract(max, min))
    return x"""

def MSE(output, output_expected, example_cardinality):
    mse = np.sum(np.subtract(output, output_expected)) / example_cardinality
    mse = np.power(mse,2)
    return mse

def input_matrix(matrix):
    #print(matrix[:, 0: matrix.shape[1] -2 ])
    return matrix[:, 0: matrix.shape[1] -1 ]
def output_matrix(matrix):
    #print(matrix[:, matrix.shape[1] -2 : matrix.shape[1] ])
    return matrix[:, matrix.shape[1] -1 : matrix.shape[1] ]