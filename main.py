from ast import ExtSlice
from Ensemble import Ensemble
from numpy import random
from Model_Selection import task
import Function
import pandas
import Task
import numpy as np
import shutil
import os
import File_names as fn
import Backpropagation as bp
import copy
Function.setPandas()

def savefigure():
    shutil.rmtree("figure_prec")
    os.makedirs("figure_prec")
    shutil.move("figure","figure_prec")
    os.makedirs("figure")

'''
##########################
# CLASSIFICATION PROBLEM #
##########################

num_epoch = 500
dim_output = 1
dim_input= 17

hidden_units=[[dim_input,3,dim_output], [dim_input,4,dim_output]]
batch_array=[169]
learning_rate_init = [0.7,0.8,0.9]
alfa = [0.7,0.8, 0.9]
v_lambda = [0]
fun = ["relu", "tanh","sigmoidal"]      
fun_out=["relu", "tanh","sigmoidal"]
weight=["random"]
early_stopping = [False]
type_learning_rate = ["fixed"]

############################
# EXECUTION CLASSIFICATION #
############################


classification = Task.Classification(fn.Monk_2_tr, fn.Monk_2_ts, num_epoch,dim_output,hidden_units,batch_array,learning_rate_init,type_learning_rate, alfa,v_lambda,
                                fun,fun_out,weight,early_stopping)

top_models = classification.startexecution()

ensamble = Ensemble(top_models, 8)
ensamble.write_result(fn.top_result_monk_1)

##############################################################################################################

'''
######################
# REGRESSION PROBLEM #
######################

num_epoch = 1
dim_output = 2
dim_input= 10
num_training = -1

hidden_units=[[dim_input, 10, 10, 10, dim_output]]
batch_array=[16]
learning_rate_init = [0.0065465, 0.00035]
alfa = [0.54]
v_lambda = [0.00001]
fun = ["sigmoidal", "tanh"]       
weight=["random"]
early_stopping = [True]
type_learning_rate = ["fixed"]
########################
# EXECUTION REGRESSION #
########################

Regression= Task.Regression(fn.ML_cup, num_epoch, dim_output,hidden_units,batch_array,learning_rate_init, type_learning_rate, alfa,v_lambda,
                                fun,weight,early_stopping,num_training)

top_models  = Regression.startexecution_k_fold()
num_training=Regression.num_training

######################################################################  RANDOMIZATION PHASE #############################################################################
ensamble = Ensemble(top_models, 8)
##############################
# HYPERPARAMETER PERTUBATION #
##############################
random_top_models=[]

#take the best model and create a new_model for each with a perturbation of all hyperparameters
for model in top_models:
    #save the hyperparameters of model
    alfa =[copy.deepcopy(model.NN.alfa)]
    v_lambda=[copy.deepcopy(model.NN.v_lambda)]
    learning_rate_init=[copy.deepcopy(model.NN.learning_rate)]
    hidden_units=[copy.deepcopy(model.NN.units)]
    batch_size=[copy.deepcopy(model.NN.batch_size)]
    
    #perturbation
    hidden_units, batch_size, learning_rate_init, alfa, v_lambda=Function.pertubation(hidden_units, batch_size, learning_rate_init, alfa, v_lambda)

    #execute perturbation on single model
    Regression = Task.Regression(fn.ML_cup, num_epoch, dim_output,hidden_units,batch_size,learning_rate_init, type_learning_rate, alfa,v_lambda,
                                fun,weight,early_stopping,num_training)

    modello=Regression.startexecution_k_fold()
    num_training=Regression.num_training
    #take the best model and 
    ensamble.insert_model(modello[0]) 

#save all the precedent figure and create a new empty folder for write new graphs
#savefigure()

#devolopment set prima del retraining
ensamble.write_result(fn.top_result_ML_cup)
#test set prima del retraining
ensamble.output_average(Regression.test_set, fn.top_result_test)

#retraing 
top_models, mee_result = Regression.retraining(ensamble.getNN())
#scrivere mee_result in un file
print("Risultati MEE sul devolpment set dopo il retraining:", mee_result)
ensamble = Ensemble(top_models, 8)

output,mse,mee=ensamble.output_average(Regression.devolopment_set, fn.top_result_test_retraing)
print("MEE ENSEMBLE devolopment_set:",mee)

#test set prima del retraining
output,mse,mee=ensamble.output_average(Regression.test_set, fn.top_result_test_retraing)
print("MEE ENSEMBLE test_set:",mee)


#BLIND TEST da fare dopo vediamo come va il retraing 
Regression.top_models = ensamble.getNN()
Regression.blind_test(fn.blind_test)
#I FILE CHE CI INTERESSANO SONO:
#TOP_RESULT_ml_CUP i migliori modelli dopo aver finito anche la random grid search cor MEE sul dev set
#GENERAL_RESULT ci sono tutti i risultati (con la random search vengono sovrascritti i precedenti)
#TOP_RESULT_TEST  i migliori modelli dopo aver finito anche la random grid search sul TEST SET
#top_result_test_retraing i migliori modelli dopo il retraining con i relativi errori sul TEST SET (gli errori del devolopment set sono stampati a video)
