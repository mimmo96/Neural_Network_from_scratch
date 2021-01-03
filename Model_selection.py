import neural_network
import numpy as np
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
                        min[1] < tmp[1]
                        min = tmp

    print(min)