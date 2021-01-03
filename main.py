import numpy as np
import csv
import leggifile
import function
import neural_network
from Model_selection import model_selection
layer=2
learning_rate=0.03
num_epoch=1000
filename = "dati.csv"
batch_size=2
alfa = 0.5
v_lambda = 0

matriceinput= leggifile.leggi(filename) #np.random.rand(100, 20)*100 
matriceinput = function.normalize_input(matriceinput)
training_set, validation_set, test_set = leggifile.divide_exaples(matriceinput)
newInput=matriceinput[0]
num_righe, num_colonne = matriceinput.shape;
output_expected = matriceinput[:, (num_colonne-2):(num_colonne)]
dim_input=np.size(newInput) - 2
nj= [[dim_input,10,2,2,0]] #np.random.randint(1,5)
#CORREGGERE MATRICEINPUT CON TR_INPUT
#training_set_output = training_set[:, 0:training_set.shape[1] -2 ]

alfa = [0.1, 0.5, 0.9]
learning_rate = [0.01, 0.05, 0.09]
v_lambda = [0.1, 0.001, 0.21]

model_selection(alfa, learning_rate, v_lambda, nj, training_set, validation_set, batch_size, num_epoch)

neural_network1=neural_network.neural_network(nj, alfa, v_lambda, learning_rate, 3)
#neural_network.mini_batch()
neural_network1.trainig(training_set, validation_set, batch_size, 100)
