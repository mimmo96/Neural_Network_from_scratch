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

def output_nn(struct_layers, x_input):
    i=0
    for layer in struct_layers:
        layer.x = x_input
        print("input layer ",i,":",layer.x)
        print("matrice peso layer ",i,":")
        print(layer.w_matrix)
        x_input = layer.output()
        i=i+1
    return x_input

def backprogation(struct_layers,num_epoch,learning_rate,output_NN,output_expected):
    for epoch in range(0,num_epoch):
        #scandisce tutti i layer presenti all'interno della rete
        for i in range(np.size(struct_layers)-1,-1,-1):
            #restituisce l'oggetto layer i-esimo
            layer=struct_layers[i]
            Di=layer.x
            for j in range(0,layer.nj):
                output=output_NN[j]
                Do=derivate_sigmoid(output)
                Dloss=der_loss(output,output_expected[j])
                gradiente=dCdW(Di,Do,Dloss)
                product=np.dot(learning_rate,gradiente)
                layer.w_matrix[:,j]=np.add(layer.w_matrix[:,j],product)
                
