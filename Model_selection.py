import neural_network
import numpy as np
from function import input_matrix, output_matrix
from ThreadPool import ThreadPool
def model_selection(vector_alfa, vector_learning_rate, vector_lambda, vectors_units, training_set, validation_set, batch_size, epochs):
    min = 0
    for alfa in vector_alfa:
        for learning_rate in vector_learning_rate:
            for v_lambda in vector_lambda:
                for units in vectors_units:
                    #aggiustare ultima componente
                    NN = neural_network.neural_network(units, alfa, v_lambda, learning_rate, np.size(units)-2) 
                    tmp = NN.trainig(training_set, validation_set, batch_size, epochs)
                    if min == 0:
                        min = tmp
                    else:
                        if tmp[1] < min[1]: 
                            min = tmp

    validation_set_input = input_matrix(validation_set)
    validation_set_output = output_matrix(validation_set)
    output_NN = np.zeros(validation_set_output.shape)
    ThreadPool(min[0], validation_set_input, 0, validation_set_input.shape[0], output_NN)
    print("best model ", output_NN)
    