import numpy as np
import read_write_file
import function
from Hold_out import Hold_out
from random import randint
import Matrix_io
import itertools
import CV_k_fold
import pandas

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
df.to_csv("result/all_models.csv")

########################
# PARAMETERS TO ANALIZE#
########################

num_epoch = 20
filename = "CUP/ML-CUP20-TR.csv"
file_name_test = "CUP/ML-CUP20-TR.csv"
dim_output = 2
problem_type="Regression"

###########################################

#read the data from the file and save it in a matrix
training_set = read_write_file.read_csv(problem_type,filename)
validation_set = read_write_file.read_csv(problem_type,file_name_test)
test_set = validation_set
dim_input=np.size(training_set[0]) - dim_output

#One hot encoding
if(problem_type=="classification"):
    training_set = function.one_hot_encoding(training_set)
    validation_set = function.one_hot_encoding(validation_set)
    test_set = function.one_hot_encoding(test_set)
    training_set = Matrix_io.Matrix_io(training_set, dim_output)
    validation_set = Matrix_io.Matrix_io(validation_set, dim_output)
    test_set = Matrix_io.Matrix_io(test_set, dim_output)
#input normalization 
if(problem_type=="Regression"):
    training_set = function.normalize_input(training_set,dim_output)
    training_set, validation_set, test_set = function.divide_exaples_hold_out(training_set, dim_output)


###################
# HYPERPARAMETERS #
###################
nj=[[dim_input,20,20,dim_output,0]]
batch_array=[16]
learning_rate = [0.0068, 0.007, 0.05]
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
top_k.write_result("result/top_k.csv")

##################################
# VERIFICARE SE TOP K è CORRETTO #
##################################
'''
#salvo 
    #stampo il risultato con il miglior modello
    NN=best_NN.best_model().NN
    best_loss_validation=best_loss_validation
    print_result(out_file,"\n************************* RESULT **************************")
    print_result(out_file,"parametri migliori:  epoch:"+str(epochs)+" batch_size:"+str(batch_size)+" alfa:"+str(NN.alfa)+ "  lamda:"+ str(NN.v_lambda)+ "  learning_rate:"
                +str(NN.learning_rate) +"  layer:"+str(NN.nj)+ " function:"+str(NN.function)+ " weight:"+str(NN.type_weight))
    print_result(out_file,"\nerrore sul validation migliore: "+str(best_loss_validation)) 

    #sui 10 migliori modelli trovati mi salvo i risultati e faccio ensamble
    save_test_model(best_NN,test_set)

'''