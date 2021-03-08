import numpy as np
import csv
import leggifile
import function
import neural_network
from Model_selection import model_selection
from random import randint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import Matrix_io

#----------------------------PARAMETRI DA ANALIZZARE----------------------

num_epoch=[100]
filename = "dati.csv"
file_name_test = "test_set.csv"
batch_array=[124]
dim_output = 1

#mi crea i layer in questo modo: (num_input, num_units_layer1, num_units_layer_2, .... , num_output, 0)
#nj=[ [10, 20, 1, 0], [10, 10, 2 , 0], [10, 30, 2 , 0],[10, 100, 2 , 0],[10, 50, 2 , 0],[10, 7, 2 , 0] ]

learning_rate = [2,4,0.1,0.44,0.01, 0.075, 0.5,0.25]
alfa = [0,0.5, 0.8]
v_lambda = [0]
fun = ["tanh"]
fun_out=["tanh"]
weight=["random"]
#-------------------------------FINE PARAMETRI------------------------------

#leggo i dati dal file e li salvo in una matrice

training_set = leggifile.leggi(filename)
validation_set = leggifile.leggi(file_name_test)
test_set = validation_set

one = OneHotEncoder(sparse=False)
label=LabelEncoder()
########################

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

#training_set = function.normalize_input(training_set,dim_output)

#divido il data set in training,validation,test
#training_set, validation_set, test_set = leggifile.divide_exaples(training_set, dim_output)
newInput=training_set[0]
dim_input=np.size(newInput) - dim_output

training_set = Matrix_io.Matrix_io(training_set, dim_output)
validation_set = Matrix_io.Matrix_io(validation_set, dim_output)
test_set = Matrix_io.Matrix_io(test_set, dim_output)


nj=[[dim_input,2, dim_output,0], [dim_input,3,dim_output,0], [dim_input,4,dim_output,0]]

model_selection(alfa, learning_rate, v_lambda, nj, training_set, validation_set,test_set, batch_array, num_epoch,fun, fun_out, weight)
