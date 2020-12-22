import numpy as np
import matplotlib.pyplot as plt
import csv
import leggifile
import normalized_initialization
import function
import Layer
import Bp

layer=2
learning_rate=0.01
num_epoch=10000
filename = "dati.csv"
batch_size=3
matriceinput=leggifile.leggi(filename)
#creo la nuova struttura che conterrà i layer
struct_layers = np.empty(layer,Layer.Layer)
matriceinput = function.normalize_input(matriceinput)
newInput=matriceinput[0]
#print(matriceinput)
num_righe, num_colonne = matriceinput.shape;
output_expected = matriceinput[:, (num_colonne-2):(num_colonne)]
#print(output_expected)
dim_input=np.size(newInput) - 2
nj= [dim_input,10,2,0] #np.random.randint(1,5)

for i in range(1,np.size(struct_layers)+1):
    struct_layers[i-1]=Layer.Layer(nj[i],nj[i-1],nj[i+1],batch_size)

#print("--------------------------------------")
#Bp.backprogation(struct_layers, num_epoch, learning_rate, newInput, output_expected)
Bp.minbetch(struct_layers,num_epoch,learning_rate,matriceinput,num_righe,batch_size,output_expected)