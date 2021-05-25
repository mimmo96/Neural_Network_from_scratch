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
    type_learning_rate = hyperparameter[2]
    function=hyperparameter[3]
    fun_out=hyperparameter[4]
    hidden=hyperparameter[5]
    learning_rate=hyperparameter[6]
    alfa=hyperparameter[7]
    v_lambda=hyperparameter[8]
    weig=hyperparameter[9]
    num_layers = np.size(hidden) - 1
    
    NN = Neural_network.Neural_network(hidden, alfa, v_lambda, learning_rate, type_learning_rate, num_layers, function, fun_out, weig, type_problem, early_stopping)
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
    print("MEE_vl ", MEE_vl)
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
    #use your cpu core for parrallelize each operation
    trials = Parallel(n_jobs=os.cpu_count(), verbose=50)(delayed(task)(type_problem, training_set,validation_set,epochs,num_training+(i/10), hyperparameter) for i in range(5))
    
    best_trial = Ensemble.Ensemble(trials, np.size(trials)).best_neural_network()
    
    return best_trial