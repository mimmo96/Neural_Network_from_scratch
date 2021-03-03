import numpy as np
import csv
import leggifile
import function
import neural_network
from Model_selection import model_selection
from random import randint


#----------------------------PARAMETRI DA ANALIZZARE----------------------

num_epoch=[100]
filename = "dati.csv"
batch_array=[1]
dim_output = 1

#mi crea i layer in questo modo: (num_input, num_units_layer1, num_units_layer_2, .... , num_output, 0)
#nj=[ [10, 20, 1, 0], [10, 10, 2 , 0], [10, 30, 2 , 0],[10, 100, 2 , 0],[10, 50, 2 , 0],[10, 7, 2 , 0] ]

learning_rate = [0.2,0.5,1, 4]
<<<<<<< HEAD
alfa = [0.5,0.65, 0.8,0.9]
v_lambda = [0.01,0]
fun=["sigmoidal", "tanh"]
weight=["uniform","random"]
=======
alfa = [0,0.65, 0.8,0.9]
v_lambda = [0,0.0001]
fun=["sigmoidal", "tanh"]
weight=["random"]
>>>>>>> parent of 6e50a4c (aggiunti grafici automatizzati)
#-------------------------------FINE PARAMETRI------------------------------

#leggo i dati dal file e li salvo in una matrice
matriceinput= leggifile.leggi(filename)

matriceinput = function.normalize_input(matriceinput,dim_output)

#divido il data set in training,validation,test
training_set, validation_set, test_set = leggifile.divide_exaples(matriceinput, dim_output)

newInput=matriceinput[0]
dim_input=np.size(newInput) - dim_output

nj=[ [dim_input, 5 ,dim_output,0],[dim_input,3,5,dim_output,0],[dim_input,3,6,dim_output,0] ]

model_selection(alfa, learning_rate, v_lambda, nj, training_set, validation_set,test_set, batch_array, num_epoch,fun,weight)
