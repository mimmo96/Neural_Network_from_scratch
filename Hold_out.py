import itertools
from read_write_file import print_result
import neural_network
import numpy as np
from Model_Selection import ThreadPool_average
import ensemble
import File_names as fn

#return best K models
def Hold_out(epochs, grid, training_set, validation_set,type_problem):
    num_training = 1
    top_k_models = ensemble.ensemble()
    out_file = open("result/resultexecution.txt","w")
   
    for hyperparameter in grid:
        
        #num_layers=np.size(units) - 2
        #calculate the average of the 5 generated models and return the loss and the MEE / accuracy
        model_stat = ThreadPool_average(type_problem, training_set, validation_set, epochs, num_training, hyperparameter)
    
        #insert model_stat in best model if it is in K top model
        top_k_models.insert_model(model_stat, type_problem)
        model_stat.write_result(fn.general_results)
        num_training += 1        

    out_file.close()
    return top_k_models

