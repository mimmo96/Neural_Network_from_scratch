import numpy as np
import csv
import leggifile
import function
import neural_network

layer=2
learning_rate=0.01
num_epoch=1000
filename = "dati.csv"
batch_size=3
alfa = 0.9
v_lambda = 0.01

matriceinput= leggifile.leggi(filename) #np.random.rand(100, 20)*100 
matriceinput = function.normalize_input(matriceinput)
newInput=matriceinput[0]
num_righe, num_colonne = matriceinput.shape;
output_expected = matriceinput[:, (num_colonne-2):(num_colonne)]
dim_input=np.size(newInput) - 2
nj= [dim_input,10,2,0] #np.random.randint(1,5)

neural_network=neural_network.neural_network(dim_input,layer,nj,alfa,v_lambda,num_epoch,learning_rate,
                                          matriceinput,num_righe,batch_size,output_expected)

neural_network.mini_batch()
