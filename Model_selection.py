import neural_network
import numpy as np
from function import LOSS

def model_selection(vector_alfa, vector_learning_rate, vector_lambda, vectors_units, training_set, validation_set,test_set, batch_array, epoch_array,fun,weight):
    
    #inizializzo i parametri che mi serviranno per salvare i dati dopo il training (per la scelta del miglior modello)
    best_min_validation = -1   
    best_learning_rate =vector_learning_rate[0]
    best_v_lambda=vector_lambda[0]
    best_units=vectors_units[0]
    best_alfa=vector_alfa[0]
    num_training=1;
    best_loss_training=-1
    best_function=1

    for batch_size in batch_array:
        for epochs in epoch_array:
            for function in fun:
                for units in vectors_units:
                    for learning_rate in vector_learning_rate:
                        for alfa in vector_alfa:
                            for v_lambda in vector_lambda:
                                for weig in weight:
                                    #salvo il numero di layer 
                                    numero_layer=np.size(units)-2
                                    #creo la neural network con i parametri passati
                                    NN = neural_network.neural_network(units, alfa, v_lambda, learning_rate, numero_layer,function,weig) 
                                    #restituisce il minimo errore della validation
                                    NN.trainig(training_set, validation_set, batch_size, epochs,num_training) 
                                    
                                    #calcolo la loss su tutto l'intero training
                                    output_NN = np.zeros(training_set.output().shape)
                                    NN.ThreadPool_Forward(training_set.input(), 0, training_set.input().shape[0], output_NN, True)
                                    loss_training = LOSS(output_NN, training_set.output(), training_set.output().shape[0],training_set.output().shape[1])
                                    print("loss:", loss_training)
                                    
                                    if best_loss_training == -1:
                                        best_loss_training = loss_training
                                        best_model = NN
                                    else:
                                        if loss_training < best_loss_training: 
                                            print("aggiorno l'errore! \n vecchio errore:", best_loss_training,
                                            "\n nuovo errore:", loss_training)
                                            best_loss_training = loss_training
                                            best_model = NN
                                            best_learning_rate =learning_rate
                                            best_v_lambda=v_lambda
                                            best_units=units
                                            best_alfa=alfa
                                            best_function=function
                                        
                                    print("------------------------------FINE TRAINING ",num_training,"-----------------------------")
                                    num_training=num_training+1        
    
    print("parametri migliori:  epoch:",epochs," batch_size:",batch_size," alfa:",best_alfa, "  lamda:", best_v_lambda, "  learning_rate:",best_learning_rate ,"  layer:",best_units, 
    " function:",best_function, " weight:",weig)
    print("\nerrore sul training migliore: ", best_loss_training) 
    
    #ricalcolo il migliore modello sul validation_set
    validation_set_input = validation_set.input()
    validation_set_output = validation_set.output()
    output_NN = np.zeros(validation_set_output.shape)
    best_model.ThreadPool_Forward(validation_set_input, 0, validation_set_input.shape[0], output_NN, True)
    loss_validation = LOSS(output_NN, validation_set_output, validation_set_output.shape[0],validation_set_output.shape[1])
    print("errore sul validation:",loss_validation)
    
    #ricalcolo il migliore modello sul test_set
    test_set_input = test_set.input()
    test_set_output = test_set.output()
    output_NN = np.zeros(test_set_output.shape)
    best_model.ThreadPool_Forward(test_set_input, 0, test_set_input.shape[0], output_NN, True)
    loss_test = LOSS(output_NN, test_set_output, test_set_output.shape[0],test_set_output.shape[1])
    print("errore sui test:",loss_test)

    return best_model



