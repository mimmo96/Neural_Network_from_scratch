import numpy as np
import matplotlib.pyplot as plt
import csv
import leggifile
import normalized_initialization
import function
import Layer
import Bp

layer=3
learning_rate=0.1
num_epoch=1000
filename = "dati.csv"
batch_size=1
matriceinput=leggifile.leggi(filename)
#creo la nuova struttura che conterrà i layer
struct_layers = np.empty(layer,Layer.Layer)

newInput=matriceinput[0]

num_righe, num_colonne = matriceinput.shape;
output_expected = matriceinput[:, (num_colonne-2):(num_colonne)]
dim_input=np.size(newInput) - 2
nj= [dim_input,3,4,2,0] #np.random.randint(1,5)

for i in range(1,np.size(struct_layers)+1):
    struct_layers[i-1]=Layer.Layer(nj[i],nj[i-1],nj[i+1],batch_size)

#print("--------------------------------------")
#Bp.backprogation(struct_layers, num_epoch, learning_rate, newInput, output_expected)
Bp.minbetch(struct_layers,num_epoch,learning_rate,matriceinput,num_righe,batch_size,output_expected)