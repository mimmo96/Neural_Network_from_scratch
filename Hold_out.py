import itertools
from read_write_file import print_result
import neural_network
import numpy as np
from function import LOSS
from Model_Selection import save,ThreadPool_average,save_test_model

def Hold_out(grid, training_set, validation_set,test_set,type_problem):
    
    #inizializzo i parametri che mi serviranno per salvare i dati dopo il training (per la scelta del miglior modello)
    best_loss_validation=-1
    MEE=0
    num_training=1
    # vettore dove salvo i migliori 10 modelli
    best_NN=np.empty(0,neural_network.neural_network)
    # file dove andrò a scrivere 
    out_file = open("result/resultexecution.txt","w")
   
    for hyperparameter in grid:
        #iperparametri
        epochs=hyperparameter[0]
        batch_size=hyperparameter[1]
        function=hyperparameter[2]
        fun_out=hyperparameter[3]
        units=hyperparameter[4]
        learning_rate=hyperparameter[5]
        alfa=hyperparameter[6]
        v_lambda=hyperparameter[7]
        weig=hyperparameter[8]

        print_result(out_file,"------------------------------INIZIO TRAINING "+str(num_training)+"-----------------------------")
        print_result(out_file,"epoch:"+str(epochs)+" batch_size:"+str(batch_size)+" alfa:"+str(alfa)+ 
        "  lamda:"+ str(v_lambda)+ "  learning_rate:"+str(learning_rate) +
        "  layer:"+str(units)+ " function:"+str(function)+ " weight:"+ str(weig))
        #salvo il numero di layer 
        numero_layer=np.size(units)-2

        #calcvolo la media dei 5 modelli generati e restituisco la loss e la MEE/accuratezza
        loss_validation,MEE,NN=ThreadPool_average(type_problem,fun_out,training_set,validation_set, batch_size, epochs,num_training,units, alfa, v_lambda, learning_rate, numero_layer,weig,function)
        #stampo il risultato di fine training
        print_result(out_file,"RESULT:") 
        print_result(out_file,"MEDIA LOSS:"+ str(loss_validation)+ " \nMEDIA MEE:"+ str(MEE))

        if best_loss_validation == -1:
            best_loss_validation = loss_validation
            best_model = NN
            best_NN=save(best_NN,NN)
        else:
            if loss_validation < best_loss_validation: 
                print_result(out_file,"aggiorno l'errore! \n vecchio errore:"+ str(best_loss_validation)+
                "\n nuovo errore:"+ str(loss_validation))
                best_loss_validation = loss_validation
                best_model = NN
                best_NN=save(best_NN,best_model)
            
                #salvo il nuovo modello nell'array

        print_result(out_file,"------------------------------FINE TRAINING "+str(num_training)+"-----------------------------")
        num_training=num_training+1        

    print_result(out_file,"\n************************* RESULT **************************")
    print_result(out_file,"parametri migliori:  epoch:"+str(epochs)+" batch_size:"+str(batch_size)+" alfa:"+str(NN.alfa)+ "  lamda:"+ str(NN.v_lambda)+ "  learning_rate:"
                +str(NN.learning_rate) +"  layer:"+str(NN.nj)+ " function:"+str(NN.function)+ " weight:"+str(NN.type_weight))
    print_result(out_file,"\nerrore sul validation migliore: "+str(best_loss_validation)) 

    #sui 10 migliori modelli trovati mi salvo i risultati e faccio ensamble
    save_test_model(best_NN,test_set)

    #chiudo i file
    out_file.close()
    return best_model

