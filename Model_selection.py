from read_write_file import print_result
import neural_network
import numpy as np
import os
import math
import ensemble
from function import LOSS, accuracy,MEE
from joblib import Parallel, delayed

##########################################
#TASK FOR COMPUTE MSE,MEE ON SINGLE MODEL#
##########################################

def task(type_problem,fun_out, NN,training_set,validation_set, batch_size, epochs,num_training):
    NN.trainig(training_set, validation_set, batch_size, epochs,num_training) 
    
    #LOSS TRAINING
    output_NN = np.zeros(training_set.output().shape)
    NN.Forwarding(training_set.input(), training_set.input().shape[0], output_NN, True)
    loss_training = LOSS(output_NN, training_set.output(), training_set.output().shape[0], penalty_term=0)

    #LOSS VALIDATION
    output_NN = np.zeros(validation_set.output().shape)
    NN.Forwarding(validation_set.input(), validation_set.input().shape[0], output_NN, True)
    loss_validation = LOSS(output_NN, validation_set.output(), validation_set.output().shape[0], penalty_term=0)

    #MEE OR ACCURACY
    if(type_problem == "classification"):
        MEE_tot = accuracy(fun_out,validation_set.output(),output_NN)
    else:
        MEE_tot = MEE(output_NN, validation_set.output(), validation_set.output().shape[0])
    
    return loss_training,loss_validation,MEE_tot,NN
    
############################################
#THREADPOOL FOR PARALLELIZATION OF 5 MODEL #
############################################
def ThreadPool_average(type_problem,fun_out,training_set,validation_set, batch_size, epochs,num_training,units, alfa, v_lambda, learning_rate, numero_layer,weig,function, early_stopping):
   
    loss_val_tot=0
    loss_tr_tot=0
    best_loss=+ math.inf
    MEE_tot=0
    best_NN=0
    loss=[]

    #create a new neuralnetwork 
    NN = neural_network.neural_network(units, alfa, v_lambda, learning_rate, numero_layer,function, fun_out, weig, type_problem, early_stopping) 
    #use your cpu core for parrallelize each operation and return the result of type [(loss1,mess1,nn1),(loss2,mess2,nn2),(loss3,mess3,nn3)]
    result= Parallel(n_jobs=os.cpu_count(), verbose=50)(delayed(task)(type_problem,fun_out, NN,training_set,validation_set, batch_size, epochs,num_training+(i/10)) for i in range(5))

    #for each model of the same neural network compute the mean of mse_tr,mse_vl,mee
    for i in range (0,5):
        loss_tr,loss_vl,mee,NN=result[i]
        loss.append(loss_vl)
        print("loss training ",i,":" ,loss_tr)
        print("loss validation ",i,":" ,loss_vl)
        print("MEE ",i,":", mee)
        if(loss_vl<best_loss):
            best_loss=loss_vl
            best_NN=NN
        #best_loss=loss_vl
        loss_val_tot = loss_val_tot + loss_vl
        loss_tr_tot = loss_tr_tot + loss_tr
        MEE_tot= MEE_tot +mee

    print("array loss:",loss)
    std=np.std(loss)
    loss_tr_tot=np.divide(loss_tr_tot,5)
    loss_val_tot=np.divide(loss_val_tot,5)
    MEE_tot=np.divide(MEE_tot,5)

    return loss_tr_tot,loss_val_tot,MEE_tot,std,best_NN


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
        neural.Forwarding(test_set.input(), 0, test_set.input().shape[0], output_NN, True)
        loss_test = LOSS(output_NN, test_set.output(), test_set.output().shape[0],0)
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