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
    hidden=hyperparameter[4]
    learning_rate=hyperparameter[5]
    alfa=hyperparameter[6]
    v_lambda=hyperparameter[7]
    weig=hyperparameter[8]
    num_layers=np.size(hidden) -1
    
    #hidden=sethidden(training_set,units)
    #print("hidden:",hidden)
    NN = Neural_network.Neural_network(hidden, alfa, v_lambda, learning_rate, num_layers, function, fun_out, weig, type_problem, early_stopping)
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
    #trials = Parallel(n_jobs=os.cpu_count(), verbose=50)(delayed(task)(type_problem, training_set,validation_set,epochs,num_training+(i/10), hyperparameter) for i in range(5))
    trials =[]

    for i in range(5):
        trials.append(task(type_problem, training_set,validation_set,epochs,num_training+(i/10), hyperparameter))
    
    best_trial = Ensemble.Ensemble(trials, np.size(trials)).best_neural_network()
    
    return best_trial

################################
# SAVE THE RESULT OF EXECUTION #
################################
def save_test_model(best_NN,test_set):
    file = open("result/best_model/result.txt","w")
    conta=1
    losses = []

    #print the results of the Neural_networks on the file
    for neural in best_NN:
        output_NN = np.zeros(test_set.output().shape)
        neural.Forwarding(test_set.input(), output_NN, True)
        loss_test = LOSS(output_NN, test_set.output(), 0)
        print_result(file,"------------------------------MODEL "+str(conta)+"-----------------------------")
        print_result(file,"PARAMETERS:  alfa:"+str(neural.alfa)+ "  lamda:"+ str(neural.v_lambda)+ "  learning_rate:"
                +str(neural.learning_rate) +"  layer:"+str(neural.units)+ " function:"+str(neural.function)+ " weight:"+str(neural.type_weight))

        print_result(file,"RESULT ON TEST SET "+str(loss_test))
        print_result(file,"--------------------------------------------------------------------------------")
        conta=conta+1
        losses.append ( loss_test )
    
    print("deviazione standard:",np.std(losses))
    print("varianza:",np.var(losses))
    
    #ensamble on test set
    en=Ensemble.Ensemble(best_NN)
    loss_Ensemble=en.loss_average(test_set)
    print_result(file,"errore sui test:" +str(loss_Ensemble)) 
    file.close()