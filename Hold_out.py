import itertools
from read_write_file import print_result
import neural_network
import numpy as np
from function import LOSS
from Model_Selection import save,ThreadPool_average,save_test_model
import ensemble

#return best K models
def Hold_out(epochs,grid, training_set, validation_set,type_problem):
    MEE = 0
    num_training = 1
    best_NN = ensemble.ensemble()
    out_file = open("result/resultexecution.txt","w")
   
    for hyperparameter in grid:
        early_stopping = hyperparameter[0]
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
        
        num_layers=np.size(units) - 2
        #calculate the average of the 5 generated models and return the loss and the MEE / accuracy
        loss_training,loss_validation,MEE,std,NN = ThreadPool_average(type_problem,fun_out,training_set,validation_set, batch_size, epochs,num_training,units, alfa, v_lambda, learning_rate, num_layers,weig,function, early_stopping)
        
        #create model_stat 
        model_stat = ensemble.stat_model(NN, loss_training, loss_validation, std, MEE, num_training)
        #insert model_stat in best model if it is in K top model
        best_NN.k_is_in_top(model_stat, type_problem)
        model_stat.write_result("result/all_models.csv")
        
        print_result(out_file,"RESULT:") 
        print_result(out_file,"MEDIA LOSS:"+ str(loss_validation)+ " \nMEDIA MEE:"+ str(MEE)+ " \nSTANDARD DEVIATION:"+ str(std))
        print_result(out_file,"RESULT TOP K:" + str(best_NN.print_top())) 
        
        print_result(out_file,"------------------------------FINE TRAINING "+str(num_training)+"-----------------------------")
        num_training=num_training+1        

    out_file.close()
    return best_NN

