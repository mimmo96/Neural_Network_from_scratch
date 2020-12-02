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
num_epoch=10
filename = "dati.csv"
matriceinput=leggifile.leggi(filename)
#creo la nuova struttura che conterrà i layer
struct_layers = np.empty(layer,Layer.Layer)
#numero di nodi

#print(matriceinput)

newInput=matriceinput[0]
newInput=np.append(newInput,1)
dim_input=np.size(newInput)
x = np.zeros(dim_input)
num_righe, num_colonne = matriceinput.shape;
output_expected = [matriceinput[0][num_colonne - 2], matriceinput[0][num_colonne - 1]]
#print(output_expected)
nj= [num_righe, 5, np.size(output_expected), 1] #np.random.randint(1,5)

for i in range(np.size(struct_layers)):
    #inizializza i nostri layer e li stampo a schermo
    struct_layers[i]=Layer.Layer(x,nj[i],nj[i+1],[np.size(x),nj[i]])
    x=np.zeros(nj[i]+1)

batch_size=3
print("--------------------------------------")
matriceinput=matriceinput
#Bp.backprogation(struct_layers, num_epoch, learning_rate, newInput, output_expected)
Bp.minbetch(struct_layers,num_epoch,learning_rate,matriceinput,num_righe,batch_size)