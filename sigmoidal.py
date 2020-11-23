import numpy as np
import matplotlib.pyplot as plt
import math


# funzione sigmoidale
def sigmoid(x):
    return np.divide(1, (1 + np.exp(-x)))


# derivata funzione sigmoidale
def derivate_sigmoid(x):
    sig = sigmoid(x)
    return np.dot(sig, np.subtract(1, sig))


# derivata loss
def der_loss(output_layer, output_expected):
    val = (output_expected - output_layer)
    return np.dot(2, val)


# funzione costo layer
def c(Di, Do, Dc):
    return 0


def output_nn(struct_layers, x_input):
    i = np.size(struct_layers)-1
    for layer in struct_layers:
        layer.x = x_input
        #print("input layer ", i, ":", layer.x)
        #print("matrice peso layer ", i, ":")
        #print(layer.w_matrix)
        if i != 0:
            x_input = np.append(layer.output(),1)
        else:
            x_input = layer.output()
        i = i - 1
    return x_input


def backprogation(struct_layers, num_epoch, learning_rate, x_input, output_expected):
    for epoch in range(0, num_epoch):
        output_NN = output_nn(struct_layers, x_input)
        # scandisce tutti i layer presenti all'interno della rete
        for i in range(np.size(struct_layers) - 1, -1, -1):
            # restituisce l'oggetto layer i-esimo
            layer = struct_layers[i]
            Di = layer.x
            # if output layer
            delta = np.zeros(layer.nj)
            if i == (np.size(struct_layers) - 1):
                for j in range(0, layer.nj):
                    der_net = derivate_sigmoid(layer.net(j))
                    # invertire output???
                    Dloss = der_loss(output_NN[j], output_expected[j])
                    delta[j] = np.dot(Dloss, der_net)
                    gradiente = np.dot(layer.x, delta[j])
                    product = np.dot(learning_rate, gradiente)
                    layer.w_matrix[:, j] = np.add(layer.w_matrix[:, j], product)
                
                print(np.subtract(output_expected, output_NN))
            else:
                for j in range(layer.nj):
                    der_net = derivate_sigmoid(layer.net(j))
                    #print("vettore pesi nodo", j, struct_layers[i+1].w_matrix[:,j],
                    #"vettore delta_out ", delta_out)
                    product = np.dot(struct_layers[i+1].w_matrix[j,:],delta_out)
                    delta[j] = np.sum(product)
                    delta[j] = np.dot(delta[j], der_net)
                    gradiente = np.dot(layer.x, delta[j])
                    product = np.dot(learning_rate, delta[j])
                    layer.w_matrix[:,j] = np.add(layer.w_matrix[:, j], product)

            delta_out = delta
    print(output_NN , output_expected)
