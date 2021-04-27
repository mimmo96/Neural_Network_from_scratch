from read_write_file import print_result
import neural_network
import numpy as np
import os
import math
import ensemble
from function import LOSS, accuracy,MEE
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
    num_layers=np.size(units) - 2

    NN = neural_network.neural_network(units, alfa, v_lambda, learning_rate, num_layers, function, fun_out, weig, type_problem, early_stopping)
    NN.trainig(training_set, validation_set, batch_size, epochs,num_training) 
    
    #LOSS TRAINING
    output_NN = np.zeros(training_set.output().shape)
    NN.Forwarding(training_set.input(), output_NN, True)
    loss_training = LOSS(output_NN, training_set.output(), penalty_term=0)

    #MEE TRAINING
    MEE_tr = MEE(output_NN, training_set.output())

    #LOSS VALIDATION
    output_NN = np.zeros(validation_set.output().shape)
    NN.Forwarding(validation_set.input(), output_NN, True)
    loss_validation = LOSS(output_NN, validation_set.output(), penalty_term=0)
    
    #MEE VALIDATION
    MEE_vl = MEE(output_NN, validation_set.output())
    
    #ACCURACY
    acc_tr = -1
    acc_vl = -1
    if(type_problem == "classification"):
        acc_tr = accuracy(fun_out,training_set.output(),output_NN)
        acc_vl = accuracy(fun_out,validation_set.output(),output_NN)
    model = ensemble.stat_model(NN, loss_training, loss_validation, MEE_tr, MEE_vl, acc_tr, acc_vl, -1, num_training)
    return model
    
############################################
#THREADPOOL FOR PARALLELIZATION OF 5 MODEL #
############################################
def ThreadPool_average(type_problem,training_set,validation_set, epochs,num_training,hyperparameter):
    
    loss_vl_tot = 0
    loss_tr_tot = 0
    MEE_tot_tr = 0
    MEE_tot_vl = 0
    acc_tot_tr = 0
    acc_tot_vl = 0 
    best_loss_validation=+ math.inf
    
    best_NN=0
    loss=[]

    #create a new neuralnetwork 
    #NN = neural_network.neural_network(units, alfa, v_lambda, learning_rate, numero_layer,function, fun_out, weig, type_problem, early_stopping) 
    #use your cpu core for parrallelize each operation and return the result of type [(loss1,mess1,nn1),(loss2,mess2,nn2),(loss3,mess3,nn3)]
    trials = Parallel(n_jobs=os.cpu_count(), verbose=50)(delayed(task)(type_problem, training_set,validation_set,epochs,num_training+(i/10), hyperparameter) for i in range(5))
    best_trial = ensemble.ensemble(trials, [], np.size(trials)).best_neural_network()
    #for each model of the same neural network compute the mean of mse_tr,mse_vl,mee
    '''
    for i in range (0,5):
        loss_tr, loss_vl, MEE_tr, MEE_vl, acc_tr, acc_vl, NN = result[i]
        loss.append(loss_vl)
        
        print("ls_tr ",i,":" ,loss_tr)
        print("ls_vl ",i,":" ,loss_vl)
        print("MEE ",i,":", MEE_tr)
        print("MEE ",i,":", MEE_vl)
        
        if(loss_vl < best_loss_validation):
            best_loss_validation = loss_vl
            best_NN = NN
        #best_loss_validation=loss_vl
        loss_vl_tot += loss_vl
        loss_tr_tot += loss_tr
        MEE_tot_tr += MEE_tr
        MEE_tot_vl += MEE_vl 
        acc_tot_tr += acc_tr
        acc_tot_vl += acc_vl

    print("array loss:",loss)
    std=np.std(loss)
    loss_tr_tot = np.divide(loss_tr_tot,5)
    loss_vl_tot = np.divide(loss_vl_tot,5)
    MEE_tot_tr = np.divide(MEE_tot_tr,5)
    MEE_tot_vl = np.divide(MEE_tot_vl,5)
    acc_tot_tr = np.divide(acc_tr,5)
    acc_tot_vl = np.divide(acc_vl,5)
    '''
    #return loss_tr_tot, loss_vl_tot, MEE_tot_tr, MEE_tot_vl, acc_tot_tr, acc_tot_vl, std, best_NN
    return best_trial

# top_model = array containing the top 10 models
# new_model = model to be inserted between 10
# each best model will be inserted in the first position and will scroll the others, if exceed the size of 10 delete the last one inserted
def save(top_model,new_model):
    top_model=np.insert(top_model,0,new_model)
    if(np.size(top_model)>10):
        top_model=np.delete(top_model,-1)
    
    return top_model

######################################
# SAVE THE RESULT OVER 10 BEST MODEL #
######################################
def save_test_model(best_NN,test_set):
    file = open("result/best_model/result.txt","w")
    conta=1
    losses = []

    #print the results of the neural_networks on the file
    for neural in best_NN:
        output_NN = np.zeros(test_set.output().shape)
        neural.Forwarding(test_set.input(), output_NN, True)
        loss_test = LOSS(output_NN, test_set.output(), 0)
        print_result(file,"------------------------------MODEL "+str(conta)+"-----------------------------")
        print_result(file,"PARAMETERS:  alfa:"+str(neural.alfa)+ "  lamda:"+ str(neural.v_lambda)+ "  learning_rate:"
                +str(neural.learning_rate) +"  layer:"+str(neural.nj)+ " function:"+str(neural.function)+ " weight:"+str(neural.type_weight))

        print_result(file,"RESULT ON TEST SET "+str(loss_test))
        print_result(file,"--------------------------------------------------------------------------------")
        conta=conta+1
        losses.append ( loss_test )
    
    print("deviazione standard:",np.std(losses))
    print("varianza:",np.var(losses))
    
    #ensamble on test set
    en=ensemble.ensemble(best_NN,test_set)
    loss_ensemble=en.loss_average()
    print_result(file,"errore sui test:" +str(loss_ensemble)) 
    file.close()