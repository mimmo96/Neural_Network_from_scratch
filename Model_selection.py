from Read_write_file import print_result
import Neural_network
import numpy as np
import os
import Ensemble
from Function import LOSS, accuracy,MEE
from joblib import Parallel, delayed

############################################
#TASK FOR COMPUTE MSE,MEE ON A SINGLE MODEL#
############################################

def task(type_problem,training_set,validation_set, epochs,num_training, hyperparameter):
    
    early_stopping = hyperparameter[0]
    batch_size=hyperparameter[1]
    function=hyperparameter[2]
    fun_out=hyperparameter[3]
    units=hyperparameter[4]
    learning_rate=hyperparameter[5]
    alfa=hyperparameter[6]
    v_lambda=hyperparameter[7]
    weig=hyperparameter[8]
    num_layers=np.size(units) - 1
    
    #units=setunits(training_set,units)
    #print("units:",units)
    NN = Neural_network.Neural_network(units, alfa, v_lambda, learning_rate, num_layers, function, fun_out, weig, type_problem, early_stopping)
    NN.trainig(training_set, validation_set, batch_size, epochs,num_training) 
    
    #LOSS TRAINING
    output_tr = np.zeros(training_set.output().shape)
    NN.Forwarding(training_set.input(), output_tr, True)
    loss_training = LOSS(output_tr, training_set.output(), penalty_term=0)

    #MEE TRAINING
    MEE_tr = MEE(output_tr, training_set.output())

    #LOSS VALIDATION
    output_vl = np.zeros(validation_set.output().shape)
    NN.Forwarding(validation_set.input(), output_vl, True)
    loss_validation = LOSS(output_vl, validation_set.output(), penalty_term=0)

    
    #MEE VALIDATION
    MEE_vl = MEE(output_vl, validation_set.output())
    print("MSE ", loss_training)
    #ACCURACY
    acc_tr = -1
    acc_vl = -1
    if(type_problem == "classification"):
        acc_tr = accuracy(fun_out,training_set.output(),output_tr)
        acc_vl = accuracy(fun_out,validation_set.output(),output_vl)
    model = Ensemble.Stat_model(NN, loss_training, loss_validation, MEE_tr, MEE_vl, acc_tr, acc_vl, -1, num_training)
    return model
    
############################################
#THREADPOOL FOR PARALLELIZATION OF 5 MODEL #
############################################
def ThreadPool_average(type_problem,training_set,validation_set, epochs,num_training,hyperparameter):
    #use your cpu core for parrallelize each operation and return the result of type [(loss1,mess1,nn1),(loss2,mess2,nn2),(loss3,mess3,nn3)]
    trials = Parallel(n_jobs=os.cpu_count(), verbose=50)(delayed(task)(type_problem, training_set,validation_set,epochs,num_training+(i/10), hyperparameter) for i in range(5))
    best_trial = Ensemble.Ensemble(trials, np.size(trials)).best_neural_network()
    print(best_trial)
    #task(type_problem, training_set,validation_set,epochs,num_training, hyperparameter)
    return best_trial
