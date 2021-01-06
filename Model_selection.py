import neural_network
import numpy as np
def model_selection(vector_alfa, vector_learning_rate, vector_lambda, vectors_units, training_set, validation_set, batch_size, epochs):
    
    best_min_validation = -1
    for alfa in vector_alfa:
        for learning_rate in vector_learning_rate:
            for v_lambda in vector_lambda:
                for units in vectors_units:
                    #aggiustare ultima componente
                    NN = neural_network.neural_network(units, alfa, v_lambda, learning_rate, np.size(units)-2) 
                    min_validation = NN.trainig(training_set, validation_set, batch_size, epochs) 
                    if best_min_validation == -1:
                        best_min_validation = min_validation
                        best_model = NN
                    else:
                        if min_validation < best_min_validation: 
                            best_min_validation = min_validation
                            best_model = NN

    validation_set_input = validation_set.input()
    validation_set_output = validation_set.output()
    output_NN = np.zeros(validation_set_output.shape)
    best_model.ThreadPool_Forward(validation_set_input, 0, validation_set_input.shape[0], output_NN, True)
    print("best model ", output_NN)



