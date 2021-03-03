from function import choose_derivate_function
import numpy as np
#CASO 1, output layer:
#type(output_expected) = value 
#type(output_expected) = value
#type(net) = value
#type(type_function) = stringa
def _delta_output_layer(output_expected, output_NN, net, type_function):
    derivata = choose_derivate_function(type_function, net)
    return derivata*(output_expected - output_NN)


#CASO 2:
#type(delta_liv_succ) = vettore contentente i delta per ogni nodo del livello successivo (un nodo = un delta)
#type(w_liv_succ)= vettore contenente peso da nodo corrente a nodo successivo ( un peso nel vettore = un nodo succesivo)
#type(net) = value
#type(type_function) = stringa
def _delta_hidden_layer(delta_liv_succ, w_liv_succ, net, type_function):
    k = np.size(delta_liv_succ) # delta_liv_succ.size == w_liv_succ
    delta_corrente = 0

    for i in range(k):
        delta_corrente += (delta_liv_succ[i]*w_liv_succ[i])
        #print("Siamo nel for di delta hidden layer")
    #print("SIAMO IN _DELTA_HIDDEN_LAYER ", delta_corrente)
    delta_corrente *= choose_derivate_function(type_function, net)
    #print("val finale delta hidden ", delta_corrente)
    return delta_corrente

#type(delta_nodo_corrente) = value 
#type(input_nodo_corrente) = vettore (output nodi precedenti: un valore = un output nodo precedente)
def gradiente(delta_nodo_corrente, input_nodo_corrente):
    #type(gradient) = vettore
    gradient = np.dot(delta_nodo_corrente, input_nodo_corrente)
    
    return gradient

#type(w_old) = pesi nodo corrente(j) (w_ji = peso per input da nodo i)
#type(gradient) = vettore
#type(learning_rate) = valore
def update_weights(w_old, learning_rate, gradient, regularizer, momentum):
    
    gradient *= learning_rate
    gradient -=regularizer
    #d_w_old = gradient
    gradient += momentum
    w_new = w_old + gradient
    
    return w_new, gradient
        