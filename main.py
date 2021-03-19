import numpy as np
import read_write_file
import function
from Hold_out import Hold_out
from random import randint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import Matrix_io
import itertools
import CV_k_fold
#----------------------------PARAMETRI DA ANALIZZARE----------------------

num_epoch = 1
filename = "CUP\ML-CUP20-TR.csv"
file_name_test = "CUP\ML-CUP20-TR.csv"
dim_output = 2
problem_type="Regression"
#mi crea i layer in questo modo: (num_input, num_units_layer1, num_units_layer_2, .... , num_output, 0)
#nj=[ [10, 20, 1, 0], [10, 10, 2 , 0], [10, 30, 2 , 0],[10, 100, 2 , 0],[10, 50, 2 , 0],[10, 7, 2 , 0] ]

learning_rate = [0.0065, 0.00001]
alfa = [0.5]
v_lambda = [0]
fun = ["tanh"]      
#fun_out non sarà considerata in caso di regressione
fun_out=["Regression"]
weight=["random"]
#-------------------------------FINE PARAMETRI------------------------------

#leggo i dati dal file e li salvo in una matrice
training_set = read_write_file.read_csv(problem_type,filename)
validation_set = read_write_file.read_csv(problem_type,filename)
test_set = validation_set

#se sono nella classificazione utilizzo one-hot-coding
if(problem_type=="classification"):
    one = OneHotEncoder(sparse=False)
    label=LabelEncoder()

    training_set_X = one.fit_transform(training_set[:,:-1])
    tmp = np.append(training_set_X,np.zeros([len(training_set_X),1]),1)
    tmp[:, -1] = label.fit_transform(training_set[:, -1])
    training_set = tmp

    validation_set_X = one.fit_transform(validation_set[:,:-1])
    tmp = np.append(validation_set_X,np.zeros([len(validation_set_X),1]),1)
    tmp[:, -1] = label.fit_transform(validation_set[:, -1])
    validation_set = tmp

    test_set_X = one.fit_transform(test_set[:,:-1])
    tmp = np.append(test_set_X,np.zeros([len(test_set_X),1]),1)
    tmp[:, -1] = label.fit_transform(test_set[:, -1])
    test_set = tmp

training_set = function.normalize_input(training_set,dim_output)

newInput=training_set[0]
dim_input=np.size(newInput) - dim_output

#cambio la divisione dei dati a seconda de lmodello che mi ritrovo ad affrontare
if(problem_type=="classification"):
    training_set = Matrix_io.Matrix_io(training_set, dim_output)
    validation_set = Matrix_io.Matrix_io(validation_set, dim_output)
    test_set = Matrix_io.Matrix_io(test_set, dim_output)

else:
    training_set, validation_set, test_set = read_write_file.divide_exaples_hold_out(training_set, dim_output)
nj=[[dim_input,20,dim_output,0]]
batch_array=[32]

#prima di fare la model selection controllo se il batch è di dimensione giusta
for i in batch_array:
    if(i>training_set.input().shape[0]):
        print("batch troppo grande!!!\nSTOPPO")
        exit()

grid = list(itertools.product(batch_array, fun, fun_out, nj, learning_rate, alfa, v_lambda,weight))

#Hold_out(grid,training_set, validation_set,test_set,problem_type)

top_k = CV_k_fold.cv_k_fold(grid, num_epoch, training_set, test_set, problem_type)
top_k.write_result(r"C:\Users\Gerlando\PycharmProjects\pythonProject1\ML_PROJECT\result\top_k.csv")
#print(top_k)