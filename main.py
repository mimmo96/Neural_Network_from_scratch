import numpy as np
import Read_write_file
import Function
from Hold_out import Hold_out
import Matrix_io
import itertools
import CV_k_fold
import pandas
import File_names as fn

df = pandas.DataFrame(columns = ['Number_Model',
                'Units_per_layer',
                'learning_rate' ,
                'lambda',
                'alfa',
                'Function_hidden',
                'inizialization_weights',
                'Error_MSE_tr',
                'Error_MSE_vl',
                'Error_MEE_tr',
                'Error_MEE_vl',
                'Accuracy_tr',
                'Accuracy_vl',
                'Variance'])

dp = pandas.DataFrame(columns = ['Number_Model',
                'Error_MSE_ts',
                'Error_MEE_ts'])

df.to_csv(fn.top_general_results)
df.to_csv(fn.general_results)
dp.to_csv(fn.top_result_test)

########################
# PARAMETERS TO ANALIZE#
########################

num_epoch = 20
dim_output = 2
problem_type="Regression"

###########################################

#read the data from the file and save it in a matrix
training_set = Read_write_file.read_csv(problem_type,fn.ML_cup)
validation_set = Read_write_file.read_csv(problem_type,fn.ML_cup)
blind_set = Read_write_file.read_csv("blind_test",fn.blind_test)
test_set = validation_set
dim_input=np.size(training_set[0]) - dim_output

#One hot encoding
if(problem_type == "classification"):
    training_set = Function.one_hot_encoding(training_set)
    validation_set = Function.one_hot_encoding(validation_set)
    test_set = Function.one_hot_encoding(test_set)
    training_set = Matrix_io.Matrix_io(training_set, dim_output)
    validation_set = Matrix_io.Matrix_io(validation_set, dim_output)
    test_set = Matrix_io.Matrix_io(test_set, dim_output)
#input normalization 
if(problem_type == "Regression"):
    training_set = Function.normalize_input(training_set,dim_output)
    training_set, validation_set, test_set = Function.divide_exaples_hold_out(training_set, dim_output)

###################
# HYPERPARAMETERS #
###################

hidden_units=[[dim_input,20,20,dim_output]]
batch_array=[16]
learning_rate = [0.054, 0.0068, 0.007, 0.088]
alfa = [0,0.5]
v_lambda = [0]
fun = ["zero_one_h"]      
fun_out=["Regression"]
weight=["Xavier Normal"]
early_stopping = [True]

for i in batch_array:
    if(i>training_set.input().shape[0]):
        print("batch.size out of bounded\n")
        exit()

###########################################
###########################################

grid = list(itertools.product(early_stopping, batch_array, fun, fun_out, hidden_units, learning_rate, alfa, v_lambda,weight))
top_model=Hold_out(num_epoch,grid,training_set, validation_set, test_set, problem_type)

##############
# BLIND TEST #
##############
count=0

output = np.zeros((blind_set.shape[0],training_set.output().shape[1]))

for model in top_model:
    output_NN = np.zeros((blind_set.shape[0],training_set.output().shape[1]))
    model.NN.Forwarding(blind_set, output_NN, True)
    output += output_NN
    count += 1
output = np.divide(output,count)

gg = pandas.DataFrame(output)
gg.to_csv(fn.result_blind_test, mode='a', header = False)

print("END")