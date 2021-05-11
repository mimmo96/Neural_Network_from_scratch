from Model_Selection import task
import Function
import Task
import numpy as np
import File_names as fn
Function.setPandas()

'''
##########################
# CLASSIFICATION PROBLEM #
##########################

num_epoch = 500
dim_output = 1
dim_input= 17

hidden_units=[[dim_input,4,dim_output]]
batch_array=[122]
learning_rate_init = [0.8]
alfa = [0.8]
v_lambda = [0,0.001,0.005,0.01,0.007]
#0.005
fun = ["relu"]      
fun_out=["sigmoidal"]
weight=["random"]
early_stopping = [False]

############################
# EXECUTION CLASSIFICATION #
############################

tot=computedim(hidden_units,batch_array,learning_rate_init,alfa,v_lambda ,fun,fun_out,weight)

classification = Task.Classification(num_epoch,dim_output,hidden_units,batch_array,learning_rate_init,alfa,v_lambda,
                                fun,fun_out,weight,early_stopping,tot)

classification.startexecution()


##############################################################################################################

'''
######################
# REGRESSION PROBLEM #
######################

num_epoch = 100
dim_output = 2
dim_input= 10

hidden_units=[ [dim_input, 20, 20, dim_output]]
batch_array=[200]
learning_rate_init = [0.0005]

alfa = [0.7]
v_lambda = [0.00001]
fun = ["leaky_relu"]      
fun_out=["Regression"]    
weight=["random"]
early_stopping = [False, True]
type_learning_rate = ["variable"]
########################
# EXECUTION REGRESSION #
########################

Regression = Task.Regression(fn.ML_cup, num_epoch, dim_output,hidden_units,batch_array,learning_rate_init, type_learning_rate, alfa,v_lambda,
                                fun,weight,early_stopping)

Regression.startexecution_k_fold()
