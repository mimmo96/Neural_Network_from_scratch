import numpy as np
import csv
import leggifile
import function
import neural_network
from Model_selection import model_selection
layer=2
learning_rate=0.01
num_epoch=1000
filename = "dati.csv"
batch_size=3
alfa = 0.5
v_lambda = 0

'''INSERIRE VARIABILE GLOBALE PER DEFINIRE NUMERO DI OUTPUT'''

matriceinput= leggifile.leggi(filename) #np.random.rand(100, 20)*100 
matriceinput = function.normalize_input(matriceinput)
training_set, validation_set, test_set = leggifile.divide_exaples(matriceinput)
newInput=matriceinput[0]
dim_input=np.size(newInput) - 1
nj= [[dim_input,5,1,0], [dim_input,10,1,0] ] #np.random.randint(1,5)
#CORREGGERE MATRICEINPUT CON TR_INPUT
#training_set_output = training_set[:, 0:training_set.shape[1] -2 ]

alfa = [0, 0.5, 0.9]
learning_rate = [0.01, 0.05, 0.1]
v_lambda = [0, 0.01, 0.1]

model_selection(alfa, learning_rate, v_lambda, nj, training_set, validation_set, batch_size, num_epoch)

#neural_network1=neural_network.neural_network(nj, alfa, v_lambda, learning_rate, 3)
#neural_network.mini_batch()
#neural_network1.trainig(training_set, validation_set, batch_size, 100)
