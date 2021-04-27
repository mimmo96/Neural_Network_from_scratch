import numpy as np
import read_write_file
import function
from Hold_out import Hold_out
from random import randint
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
                'function_hidden',
                'inizialization_weights',
                'Error_MSE_tr',
                'Error_MSE_vl',
                'Error_MEE_tr',
                'Error_MEE_vl',
                'Accuracy_tr',
                'Accuracy_vl',
                'Variance'])
df.to_csv(fn.general_results)

########################
# PARAMETERS TO ANALIZE#
########################

num_epoch = 5
dim_output = 2
problem_type="Regression"

###########################################

#read the data from the file and save it in a matrix
training_set = read_write_file.read_csv(problem_type,fn.ML_cup)
validation_set = read_write_file.read_csv(problem_type,fn.ML_cup)
test_set = validation_set
dim_input=np.size(training_set[0]) - dim_output

#One hot encoding
if(problem_type == "classification"):
    training_set = function.one_hot_encoding(training_set)
    validation_set = function.one_hot_encoding(validation_set)
    test_set = function.one_hot_encoding(test_set)
    training_set = Matrix_io.Matrix_io(training_set, dim_output)
    validation_set = Matrix_io.Matrix_io(validation_set, dim_output)
    test_set = Matrix_io.Matrix_io(test_set, dim_output)
#input normalization 
if(problem_type == "Regression"):
    training_set = function.normalize_input(training_set,dim_output)
    training_set, validation_set, test_set = function.divide_exaples_hold_out(training_set, dim_output)


###################
# HYPERPARAMETERS #
###################
nj=[[dim_input,20,20,dim_output,0]]
batch_array=[16]
learning_rate = [1, 0.0068, 0.007, 0.05]
alfa = [0.5]
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

grid = list(itertools.product(early_stopping, batch_array, fun, fun_out, nj, learning_rate, alfa, v_lambda,weight))
top_k=Hold_out(num_epoch,grid,training_set, validation_set,problem_type)
top_k.write_result(fn.top_general_results)
