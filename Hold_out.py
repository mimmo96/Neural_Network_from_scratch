import itertools
from read_write_file import print_result
import neural_network
import numpy as np
from function import LOSS
from Model_selection import save,ThreadPool_average,save_test_model
import ensemble
def Hold_out(grid, training_set, validation_set,test_set,type_problem):
    
    #inizializzo i parametri che mi serviranno per salvare i dati dopo il training (per la scelta del miglior modello)
    best_loss_validation=-1
    MEE=0
    num_training=1
    # vettore dove salvo i migliori 10 modelli
    best_NN = ensemble.ensemble()
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
        
        #create model_stat 
        model_stat = ensemble.stat_model(NN, loss_validation, MEE, num_training)
        #insert model_stat in best model if it is in K top model
        best_NN.k_is_in_top(model_stat)

###########################################AGGIORNA MIMMO PRO################################################################

        #stampo il risultato di fine training
        print_result(out_file,"RESULT:") 
        print_result(out_file,"MEDIA LOSS:"+ str(loss_validation)+ " \nMEDIA MEE:"+ str(MEE))
        
        print_result(out_file,"RESULT TOP K:" + str(best_NN.print_top())) 
        

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

