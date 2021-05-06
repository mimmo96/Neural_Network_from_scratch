from Model_Selection import task
import Function
import Task
import numpy as np

Function.setPandas()

def computedim(hidden_units,batch_array,learning_rate,alfa,v_lambda ,fun,fun_out,weight ):
    return np.size(hidden_units,0)* np.size(batch_array,0)* np.size(learning_rate,0)* np.size(alfa,0)* np.size(v_lambda,0)* np.size(fun,0)* np.size(fun_out,0)* np.size(weight,0)


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

num_epoch = 500
dim_output = 2
dim_input= 10

hidden_units=[[dim_input,12,dim_output]]
batch_array=[16]
learning_rate = [0.003154814]
alfa = [0.6]
v_lambda = [0]
fun = ["zero_one_h"]      
fun_out=["Regression"]    
weight=["Xavier Normal"]
early_stopping = [False]

########################
# EXECUTION REGRESSION #
########################

tot=computedim(hidden_units,batch_array,learning_rate,alfa,v_lambda ,fun,fun_out,weight)

Regression = Task.Regression(num_epoch,dim_output,hidden_units,batch_array,learning_rate,alfa,v_lambda,
                                fun,fun_out,weight,early_stopping,tot)

Regression.startexecution()
