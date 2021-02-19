import numpy as np
import csv
import leggifile
import function
import neural_network
from Model_selection import model_selection
from random import randint


#----------------------------PARAMETRI DA ANALIZZARE----------------------

num_epoch=[100,500]
filename = "dati.csv"
batch_array=[1,2,4,8,16,32]
dim_output = 2

#mi crea i layer in questo modo: (num_input, num_units_layer1, num_units_layer_2, .... , num_output, 0)
nj=[ [10, 20, 2, 0], [10, 10, 2 , 0], [10, 30, 2 , 0],[10, 100, 2 , 0],[10, 50, 2 , 0],[10, 7, 2 , 0] ]
learning_rate = [0.001,0.002,0.2,0.5,0.007, 1, 2, 3, 4]
alfa = [0,0.5,0.6,0.7,0.8,0.9]
v_lambda = [0.01,0]
fun=["sigmoidal","tanh","relu"]
weight=["uniform","random","Xavier Normal","Xavier Uniform","He uniform","He Normal"]
#-------------------------------FINE PARAMETRI------------------------------

#leggo i dati dal file e li salvo in una matrice
matriceinput= leggifile.leggi(filename)
matriceinput = function.normalize_input(matriceinput,dim_output)
#divido il data set in training,validation,test
training_set, validation_set, test_set = leggifile.divide_exaples(matriceinput, dim_output)

newInput=matriceinput[0]
dim_input=np.size(newInput) - dim_output

model_selection(alfa, learning_rate, v_lambda, nj, training_set, validation_set,test_set, batch_array, num_epoch,fun,weight)