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
hidden_layer=2
learning_rate=[0.01]
num_epoch=1000
filename = "dati.csv"
batch_size=3
alfa = [0.5]
v_lambda = [0]
dim_output = 1

matriceinput= leggifile.leggi(filename) #np.random.rand(100, 20)*100 
matriceinput = function.normalize_input(matriceinput)
training_set, validation_set, test_set = leggifile.divide_exaples(matriceinput, dim_output)

newInput=matriceinput[0]
dim_input=np.size(newInput) - dim_output
#nj= set_hiddenunits(hidden_layer,dim_input,dim_output)
nj=[[dim_input,10, 10, dim_output,0], [dim_input,15,10,dim_output,0] ]
#CORREGGERE MATRICEINPUT CON TR_INPUT
#training_set_output = training_set[:, 0:training_set.shape[1] -2 ]

alfa = [0, 0.5, 0.9]
learning_rate = [0.001, 0.005, 0.01]
v_lambda = [0, 0.01, 0.05, 0.07]

model_selection(alfa, learning_rate, v_lambda, nj, training_set, validation_set, batch_size, num_epoch)

#neural_network1=neural_network.neural_network(nj, alfa, v_lambda, learning_rate, 3)
#neural_network.mini_batch()
#neural_network1.trainig(training_set, validation_set, batch_size, 100)
