import itertools
from read_write_file import print_result
import neural_network
import numpy as np
from function import LOSS
from Model_Selection import save,ThreadPool_average,save_test_model
import ensemble

def Hold_out(epochs,grid, training_set, validation_set,type_problem):
    #inizializzo i parametri che mi serviranno per salvare i dati dopo il training (per la scelta del miglior modello)
    MEE=0
    num_training=1
    # vettore dove salvo i migliori 10 modelli
    best_NN = ensemble.ensemble()
    # file dove andrò a scrivere 
    out_file = open("result/resultexecution.txt","w")
   
    for hyperparameter in grid:
        #iperparametri
        batch_size=hyperparameter[0]
        function=hyperparameter[1]
        fun_out=hyperparameter[2]
        units=hyperparameter[3]
        learning_rate=hyperparameter[4]
        alfa=hyperparameter[5]
        v_lambda=hyperparameter[6]
        weig=hyperparameter[7]

        print_result(out_file,"------------------------------INIZIO TRAINING "+str(num_training)+"-----------------------------")
        print_result(out_file,"epoch:"+str(epochs)+" batch_size:"+str(batch_size)+" alfa:"+str(alfa)+ 
        "  lamda:"+ str(v_lambda)+ "  learning_rate:"+str(learning_rate) +
        "  layer:"+str(units)+ " function:"+str(function)+ " weight:"+ str(weig))
        #salvo il numero di layer 
        numero_layer=np.size(units)-2

        #calcvolo la media dei 5 modelli generati e restituisco la loss e la MEE/accuratezza
        loss_training,loss_validation,MEE,std,NN=ThreadPool_average(type_problem,fun_out,training_set,validation_set, batch_size, epochs,num_training,units, alfa, v_lambda, learning_rate, numero_layer,weig,function)
        
        #create model_stat 
        model_stat = ensemble.stat_model(NN, loss_training, loss_validation, std, MEE, num_training)
        #insert model_stat in best model if it is in K top model
        best_NN.k_is_in_top(model_stat, type_problem)
        model_stat.write_result("result/all_models.csv")
        #stampo il risultato di fine training
        print_result(out_file,"RESULT:") 
        print_result(out_file,"MEDIA LOSS:"+ str(loss_validation)+ " \nMEDIA MEE:"+ str(MEE)+ " \nSTANDARD DEVIATION:"+ str(std))
        print_result(out_file,"RESULT TOP K:" + str(best_NN.print_top())) 
        
        print_result(out_file,"------------------------------FINE TRAINING "+str(num_training)+"-----------------------------")
        num_training=num_training+1        

    #chiudo i file
    out_file.close()

    #restituisco i migliori 10 modelli 
    return best_NN

