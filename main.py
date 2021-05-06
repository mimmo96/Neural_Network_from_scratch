from Model_Selection import task
import Function
import Task

Function.setPandas()

##########################
# CLASSIFICATION PROBLEM #
##########################

num_epoch = 500
dim_output = 1
dim_input= 17

hidden_units=[[dim_input,2,dim_output],[dim_input,3,dim_output],[dim_input,4,dim_output]]
batch_array=[16]
learning_rate = [0.44,0.5,0.8]
alfa = [0.8,0.9]
v_lambda = [0]
fun = ["relu","sigmoidal","tanh"]      
fun_out=["sigmoidal","tanh"]
weight=["random","uniform"]
early_stopping = [True]

############################
# EXECUTION CLASSIFICATION #
############################

classification = Task.Classification(num_epoch,dim_output,hidden_units,batch_array,learning_rate,alfa,v_lambda,
                                fun,fun_out,weight,early_stopping)

classification.startexecution()


##############################################################################################################

'''
######################
# REGRESSION PROBLEM #
######################

num_epoch = 500
dim_output = 2
dim_input= 10

hidden_units=[[dim_input,20,20,dim_output]]
batch_array=[16]
learning_rate = [0.054, 0.0068, 0.007, 0.05]
alfa = [0.5, 0.9]
v_lambda = [0]
fun = ["zero_one_h"]      
fun_out=["Regression"]    
weight=["random"]
early_stopping = [True]

########################
# EXECUTION REGRESSION #
########################

Regression = Task.Regression(num_epoch,dim_output,hidden_units,batch_array,learning_rate,alfa,v_lambda,
                                fun,fun_out,weight,early_stopping)

Regression.startexecution()
'''
