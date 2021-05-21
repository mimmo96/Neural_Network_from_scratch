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
Function.setPandas()

def savefigure():
    shutil.rmtree("figure_prec")
    os.makedirs("figure_prec")
    shutil.move("figure","figure_prec")
    os.makedirs("figure")


##########################
# CLASSIFICATION PROBLEM #
##########################

num_epoch = 500
dim_output = 1
dim_input= 17

hidden_units=[[dim_input,3,dim_output], [dim_input,4,dim_output]]
batch_array=[124]
learning_rate_init = [0.44, 0.65, 0.8]
alfa = [0.65, 0.8, 0.9]
v_lambda = [0]
#0.005
fun = ["tanh", "sigmoidal"]      
fun_out=["sigmoidal", "relu"]
weight=["random"]
early_stopping = [True]
type_learning_rate = ["fixed"]

############################
# EXECUTION CLASSIFICATION #
############################


classification = Task.Classification(fn.Monk_1_tr, fn.Monk_1_ts, num_epoch,dim_output,hidden_units,batch_array,learning_rate_init,type_learning_rate, alfa,v_lambda,
                                fun,fun_out,weight,early_stopping)

top_models = classification.startexecution()

ensamble = Ensemble(top_models, 8)
ensamble.write_result(fn.top_result_monk_1)

##############################################################################################################

'''
######################
# REGRESSION PROBLEM #
######################

num_epoch = 500
dim_output = 2
dim_input= 10

hidden_units=[[dim_input, 10, 10, 10, dim_output]]
batch_array=[256]
learning_rate_init = [0.0065465, 0.00035]
alfa = [0.54]
v_lambda = [0.00001]
fun = ["sigmoidal", "tanh"]      
fun_out=["Regression"]    
weight=["random"]
early_stopping = [True]
type_learning_rate = ["fixed"]
########################
# EXECUTION REGRESSION #
########################

Regression = Task.Regression(fn.ML_cup, num_epoch, dim_output,hidden_units,batch_array,learning_rate_init, type_learning_rate, alfa,v_lambda,
                                fun,weight,early_stopping)

top_models = Regression.startexecution_Hold_out()


##############################
# HYPERPARAMETER PERTUBATION #
##############################
random_hidden_units=[]
random_batch_array=[]
random_learning_rate_init = []
random_alfa = []
random_v_lambda = []

for model in top_models:
    if not (model.NN.units in random_hidden_units):
        random_hidden_units.append(model.NN.units)
    if not (model.NN.batch_size in random_batch_array):
        random_batch_array.append(model.NN.batch_size)
    if not (model.NN.learning_rate_init in random_learning_rate_init):
        random_learning_rate_init.append(model.NN.learning_rate_init)
    if not (model.NN.alfa in random_alfa):
        random_alfa.append(model.NN.alfa)
    if not (model.NN.v_lambda in random_v_lambda):
        random_v_lambda.append(model.NN.v_lambda)

random_hidden_units,random_batch_array,random_learning_rate_init,random_alfa,random_v_lambda=Function.pertubation(random_hidden_units,random_batch_array,random_learning_rate_init,random_alfa,random_v_lambda)
Random_Regression = Task.Regression(fn.ML_cup, num_epoch, dim_output, random_hidden_units, random_batch_array,
                                    random_learning_rate_init, type_learning_rate, random_alfa, random_v_lambda,
                                    fun, weight, early_stopping)

#save all the precedent figure and create a new empty folder for write new graphs
savefigure()

random_top_models = Random_Regression.startexecution_k_fold()


############
# ENSAMBLE #
############
ensamble = Ensemble(top_models, 8)
for rnd_model in random_top_models:
    ensamble.insert_model(rnd_model)


#devolopment set prima del retraining
ensamble.write_result(fn.top_result_ML_cup)
#test set prima del retraining
ensamble.output_average(Regression.test_set, fn.top_result_test)


#retraing 
top_models, mee_result = Regression.retraining(ensamble.getNN())
#scrivere mee_result in un file
print("Risultati MEE sul devolpment set dopo il retraining:", mee_result)
ensamble = Ensemble(top_models, 8)

output,mee=ensamble.output_average(Regression.devolopment_set, fn.top_result_test_retraing)
print("MEE ENSEMBLE devolopment_set:",mee)

#test set prima del retraining
output,mee=ensamble.output_average(Regression.test_set, fn.top_result_test_retraing)
print("MEE ENSEMBLE test_set:",mee)


#BLIND TEST da fare dopo vediamo come va il retraing 
Regression.top_models = ensamble.getNN()
Regression.blind_test(fn.blind_test)
#I FILE CHE CI INTERESSANO SONO:
#TOP_RESULT_ml_CUP i migliori modelli dopo aver finito anche la random grid search cor MEE sul dev set
#GENERAL_RESULT ci sono tutti i risultati (con la random search vengono sovrascritti i precedenti)
#TOP_RESULT_TEST  i migliori modelli dopo aver finito anche la random grid search sul TEST SET
#top_result_test_retraing i migliori modelli dopo il retraining con i relativi errori sul TEST SET (gli errori del devolopment set sono stampati a video)
'''