import neural_network
import numpy as np

def model_selection(vector_alfa, vector_learning_rate, vector_lambda, vectors_units, training_set, validation_set,test_set, batch_size, epochs):
    
    #inizializzo i parametri che mi serviranno per salvare i dati dopo il training (per la scelta del miglior modello)
    best_min_validation = -1   
    best_learning_rate =vector_learning_rate[0]
    best_v_lambda=vector_lambda[0]
    best_units=vectors_units[0]
    best_alfa=vector_alfa[0]
    loss=0
    best_loss=0;
    num_training=1;

    for learning_rate in vector_learning_rate:
        for v_lambda in vector_lambda:
            for units in vectors_units:
                for alfa in vector_alfa:
                    #salvo il numero di layer 
                    numero_layer=np.size(units)-2
                    #creo la neural network con i parametri passati
                    NN = neural_network.neural_network(units, alfa, v_lambda, learning_rate, numero_layer) 
                    #restituisce il minimo errore della validation
                    min_validation,loss = NN.trainig(training_set, validation_set, batch_size, epochs,num_training) 
                    num_training=num_training+1
                    if best_min_validation == -1:
                        best_min_validation = min_validation
                        best_model = NN
                    else:
                        if min_validation < best_min_validation: 
                            print("aggiorno l'errore! \n vecchio errore:", best_min_validation,
                            "\n nuovo errore:", min_validation,)
                            best_min_validation = min_validation
                            best_model = NN
                            best_learning_rate =learning_rate
                            best_v_lambda=v_lambda
                            best_units=units
                            best_alfa=alfa
                            best_loss=loss
    
    print("parametri migliori:  learning_rate:",best_learning_rate, "v_lambda:",best_v_lambda,"units:",best_units,"alfa:",best_alfa)
    print("\nerrore sul training migliore: ", best_min_validation) 
    print("loss errore migliore: ", loss)
    
    #ricalcolo il migliore modello sul validation_set

    validation_set_input = validation_set.input()
    validation_set_output = validation_set.output()
    output_NN = np.zeros(validation_set_output.shape)
    best_model.ThreadPool_Forward(validation_set_input, 0, validation_set_input.shape[0], output_NN, True)
    #print("output previsto: ", validation_set_output)
    #print("best model ", output_NN)
    print("errore sul validation:",np.sum( abs(np.subtract(output_NN, validation_set_output)))/validation_set_input.shape[0])
    
    #ricalcolo il migliore modello sul test_set

    test_set_input = test_set.input()
    test_set_output = test_set.output()
    output_NN = np.zeros(test_set_output.shape)
    best_model.ThreadPool_Forward(test_set_input, 0, test_set_input.shape[0], output_NN, True)

    print("errore sui test:",np.sum( abs(np.subtract(output_NN, test_set_output)))/test_set_input.shape[0])

    return best_model



