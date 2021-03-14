import neural_network
import numpy as np
from function import LOSS, accuracy,MEE
from concurrent.futures import ThreadPoolExecutor

def model_selection(vector_alfa, vector_learning_rate, vector_lambda, vectors_units, training_set, validation_set,test_set, batch_array, epoch_array,fun,fun_output, weight,type_problem):
    
    #inizializzo i parametri che mi serviranno per salvare i dati dopo il training (per la scelta del miglior modello)
    best_learning_rate =vector_learning_rate[0]
    best_v_lambda=vector_lambda[0]
    best_units=vectors_units[0]
    best_alfa=vector_alfa[0]
    num_training=1;
    best_loss_validation=-1
    best_function=1
    MEE=0
    # vettore dove salvo i migliori 10 modelli
    best_NN=np.empty(10,neural_network.neural_network)

    for epochs in epoch_array:
        for batch_size in batch_array:
            for function in fun:
                for fun_out in fun_output:
                    for units in vectors_units:
                        for learning_rate in vector_learning_rate:
                            for alfa in vector_alfa:
                                for v_lambda in vector_lambda:
                                    for weig in weight:
                                        print("------------------------------INIZIO TRAINING ",num_training,"-----------------------------")
                                        print("epoch:",epochs," batch_size:",batch_size," alfa:",alfa, 
                                        "  lamda:", v_lambda, "  learning_rate:",learning_rate ,
                                        "  layer:",units, " function:",function, " weight:", weig)
                                        #salvo il numero di layer 
                                        numero_layer=np.size(units)-2
                                        #creo la neural network con i parametri passati
                                        NN = neural_network.neural_network(units, alfa, v_lambda, learning_rate, numero_layer,function, fun_out, weig, type_problem) 
                               
                                        #calcvolo la media dei 5 modelli generati e restituisco la loss e la MEE/accuratezza
                                        loss_validation,MEE=ThreadPool_average(type_problem,fun_out,NN,training_set,validation_set, batch_size, epochs,num_training)
                                        
                                        #stampo il risultato di fine training
                                        print("RESULT:") 
                                        print("MEDIA LOSS:", loss_validation, " \nMEDIA MEE:", MEE)
    
                                        if best_loss_validation == -1:
                                            best_loss_validation = loss_validation
                                            best_model = NN
                                            best_NN=save(best_NN,NN)
                                        else:
                                            if loss_validation < best_loss_validation: 
                                                print("aggiorno l'errore! \n vecchio errore:", best_loss_validation,
                                                "\n nuovo errore:", loss_validation)
                                                best_loss_validation = loss_validation
                                                best_model = NN
                                                best_NN=save(best_NN,best_model)
                                                best_learning_rate =learning_rate
                                                best_v_lambda=v_lambda
                                                best_units=units
                                                best_alfa=alfa
                                                best_function=function
                                                #salvo il nuovo modello nell'array

                                        print("------------------------------FINE TRAINING ",num_training,"-----------------------------")
                                        num_training=num_training+1        
    
    print("\n************************* RESULT **************************")
    print("parametri migliori:  epoch:",epochs," batch_size:",batch_size," alfa:",best_alfa, "  lamda:", best_v_lambda, "  learning_rate:",best_learning_rate ,"  layer:",best_units, 
    " function:",best_function, " weight:",weig)
    print("\nerrore sul validation migliore: ", best_loss_validation) 

    #ricalcolo il migliore modello sul test_set
    test_set_input = test_set.input()
    test_set_output = test_set.output()
    output_NN = np.zeros(test_set_output.shape)
    best_model.ThreadPool_Forward(test_set_input, 0, test_set_input.shape[0], output_NN, True)
    penalty_term = NN.penalty_NN()
    loss_test = LOSS(output_NN, test_set_output, test_set_output.shape[0],penalty_term)
    print("errore sui test:",loss_test)

    return best_model

# funzioni per parallelizzare il calcolo con i 5 modelli in modo da fare la media finale
def task(type_problem,fun_out, NN,training_set,validation_set, batch_size, epochs,num_training):
    NN.trainig(training_set, validation_set, batch_size, epochs,num_training) 
    penalty_term = NN.penalty_NN()
    output_NN = np.zeros(validation_set.output().shape)
    NN.ThreadPool_Forward(validation_set.input(), 0, validation_set.input().shape[0], output_NN, True)
    loss_tot = LOSS(output_NN, validation_set.output(), validation_set.output().shape[0], penalty_term)
    if(type_problem=="classification"):
        MEE_tot=accuracy(type_problem,fun_out,validation_set.output(),output_NN)
    else:
        MEE_tot=MEE(output_NN, validation_set.output(), validation_set.output().shape[0])
    
    return loss_tot,MEE_tot
    

def ThreadPool_average(type_problem,fun_out,NN,training_set,validation_set, batch_size, epochs,num_training):
    #creo il pool di thread
    executor = ThreadPoolExecutor(5)
    loss_tot=0
    MEE_tot=0

    for i in range(0,5):
        loss,MEE=executor.submit(task,type_problem,fun_out, NN,training_set,validation_set, batch_size, epochs,num_training).result()
        print("loss ",i,":" ,loss)
        print("MEE ",i,":", MEE)
        loss_tot = loss_tot + loss
        MEE_tot= MEE_tot +MEE

    executor.shutdown(True)

    loss_tot=np.divide(loss_tot,5)
    MEE_tot=np.divide(MEE_tot,5)

    return loss_tot,MEE_tot


# top_model=array contente i migliori 10 modelli
# new_model= modello da inserire tra i 10
# ogni migliori modello verrà inserito nella prima posizione e farà scorrerre gli altri, se supero la dimensione di 10 cancello l'ultimo inserito 
def save(top_model,new_model):
    top_model=np.insert(top_model,0,new_model)
    if(np.size(top_model)>10):
        top_model=np.delete(top_model,-1)
    
    return top_model
