import numpy as np
import csv
import leggifile
import function
import neural_network
from Model_selection import model_selection
from random import randint

'''
hidden_layer=3
dim_input=13
dim_output=1

num_hidden_units=(np.dot(dim_input,2) //3) + dim_output
nj=[]

for k in range(1,(dim_input*2)-1):
    arr=[dim_input]
    
    for i in range(hidden_layer-1):
        arr.append(k)
    
    arr.append(dim_output)
    arr.append(0)
    nj.append(arr)
print(nj)

#np.random.randint(1,5)

'''
#----------------------------PARAMETRI DA ANALIZZARE----------------------

num_epoch=500
filename = "dati.csv"
batch_size=30
dim_output = 2

#mi crea i layer in questo modo: (num_input, num_units_layer1, num_units_layer_2, .... , num_output, 0)
nj=[ [10, 15, 15, 2, 0],[10, 10, 10, 2, 0], [10, 20, 2, 0], [10, 50, 2, 0] ,[10, 100, 15, 2, 0], [10, 50, 50, 2, 0] ]
alfa = [0,0.9]
learning_rate = [0.001, 0.1,0.01]
v_lambda = [0, 0.05]

#-------------------------------FINE PARAMETRI------------------------------

#leggo i dati dal file e li salvo in una matrice
matriceinput= leggifile.leggi(filename)
#divido il data set in training,validation,test
training_set, validation_set, test_set = leggifile.divide_exaples(matriceinput, dim_output)

newInput=matriceinput[0]
dim_input=np.size(newInput) - dim_output

model_selection(alfa, learning_rate, v_lambda, nj, training_set, validation_set,test_set, batch_size, num_epoch)