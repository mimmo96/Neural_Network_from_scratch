import numpy as np
import csv
import leggifile
import function
import neural_network
from Model_selection import model_selection

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

num_epoch=100
filename = "dati.csv"
batch_size=1
dim_output = 1

matriceinput= leggifile.leggi(filename)
matriceinput = function.normalize_input(matriceinput)
training_set, validation_set, test_set = leggifile.divide_exaples(matriceinput, dim_output)

newInput=matriceinput[0]
dim_input=np.size(newInput) - dim_output

nj=[[dim_input,9, 9, dim_output,0] ]
alfa = [0, 0.5, 0.9]
learning_rate = [0.05, 0.1, 0.001, 0.9]
v_lambda = [0, 0.01, 0.05, 0.07]

model_selection(alfa, learning_rate, v_lambda, nj, training_set, validation_set, batch_size, num_epoch)