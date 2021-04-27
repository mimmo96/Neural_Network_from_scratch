from function import choose_derivate_function
import numpy as np
import warnings

######################
# DELTA OUTPUT LAYER #
######################

#type(output_expected) = value 
#type(output_expected) = value
#type(net) = value
#type(type_function) = string
def _delta_output_layer(output_expected, output_NN, net, type_function):
    derivata = choose_derivate_function(type_function, net)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  
        return -derivata*(output_expected - output_NN)


######################
# DELTA HIDDEN LAYER #
######################

#type(delta_liv_succ) = vector containing the deltas for each node of the next level (one node = one delta)
#type(w_liv_succ)= vector containing weight from current node to next node (one weight in the vector = one next node)
#type(net) = value
#type(type_function) = string
def _delta_hidden_layer(delta_liv_succ, w_liv_succ, net, type_function):
    np.seterr(all='ignore') 
    delta_coorente = np.dot(delta_liv_succ, w_liv_succ)
    derivata_corrente = choose_derivate_function(type_function, net)
    delta_coorente = np.multiply(delta_coorente, derivata_corrente)
    
    return delta_coorente
    
#############
# GRADIENT #
#############

#type(delta_nodo_corrente) = value 
#type(input_nodo_corrente) = vector (previous node output: one value = one previous node output)
def gradiente(delta_nodo_corrente, input_nodo_corrente):
        np.seterr(all='ignore') 
        gradient = np.dot(delta_nodo_corrente, input_nodo_corrente)
        return gradient

#################
# WEIGHT UPDATE #
#################

#type(w_old) = current node weights (j) (w_ji = weight for input from node i)
#type(gradient) = vector
#type(learning_rate) = value
def update_weights(w_old, learning_rate, gradient, regularizer, momentum):
    
    gradient *= learning_rate
    gradient -=regularizer
    #d_w_old = gradient
    gradient += momentum
    w_new = w_old + gradient
    
    return w_new, gradient
        