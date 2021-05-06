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
learning_rate = [0.8]
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

tot=computedim(hidden_units,batch_array,learning_rate,alfa,v_lambda ,fun,fun_out,weight)

classification = Task.Classification(num_epoch,dim_output,hidden_units,batch_array,learning_rate,alfa,v_lambda,
                                fun,fun_out,weight,early_stopping,tot)

classification.startexecution()


##############################################################################################################

'''
######################
# REGRESSION PROBLEM #
######################

num_epoch = 10
dim_output = 2
dim_input= 10

hidden_units=[[dim_input, 6, dim_output], [dim_input,12,dim_output], [dim_input, 25, dim_output], 
                [dim_input, 6, 13, dim_output], [dim_input, 10, 10, dim_output], [dim_input, 25, 25, dim_output]]
batch_array=[16, 32, 124]
learning_rate = [0.04, 0.00345, 0.00075, 0.15]

alfa = [0.5, 0.8]
v_lambda = [0.0001, 0.00001]
fun = ["sigmoidal", "tanh"]      
fun_out=["Regression"]    
weight=["random", "uniform", "Xavier Normal"]
early_stopping = [False, True]

########################
# EXECUTION REGRESSION #
########################

Regression = Task.Regression(fn.ML_cup, num_epoch, dim_output,hidden_units,batch_array,learning_rate,alfa,v_lambda,
                                fun,weight,early_stopping)

Regression.startexecution_k_fold()
