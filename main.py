import numpy as np
import csv
import leggifile
import function
import neural_network
from Model_selection import model_selection
from random import randint


#----------------------------PARAMETRI DA ANALIZZARE----------------------

num_epoch=[500]
filename = "dati.csv"
batch_array=[3, 20, 62]
dim_output = 1

#mi crea i layer in questo modo: (num_input, num_units_layer1, num_units_layer_2, .... , num_output, 0)
#nj=[ [10, 20, 1, 0], [10, 10, 2 , 0], [10, 30, 2 , 0],[10, 100, 2 , 0],[10, 50, 2 , 0],[10, 7, 2 , 0] ]

learning_rate = [0.44]
alfa = [0.44, 0.55]
v_lambda = [0]
fun = ["sigmoidal"]
fun_out=["tanh", "sigmoidal"]
weight=["uniform", "random"]
#-------------------------------FINE PARAMETRI------------------------------

#leggo i dati dal file e li salvo in una matrice
matriceinput= leggifile.leggi(filename)

#matriceinput = function.normalize_input(matriceinput,dim_output)

#divido il data set in training,validation,test
training_set, validation_set, test_set = leggifile.divide_exaples(matriceinput, dim_output)

newInput=matriceinput[0]
dim_input=np.size(newInput) - dim_output

nj=[[dim_input,2, dim_output,0], [dim_input,3,dim_output,0], [dim_input,4,dim_output,0]]

model_selection(alfa, learning_rate, v_lambda, nj, training_set, validation_set,test_set, batch_array, num_epoch,fun, fun_out, weight)
