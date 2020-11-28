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
output_expected=[20]
num_epoch=100
filename = "dati.csv"
input_value=leggifile.leggi(filename)
#creo la nuova struttura che conterrà i layer
struct_layers = np.empty(layer,Layer.Layer)
#numero di nodi
nj= [5, 5, 1,1]#np.random.randint(1,5)
#input_value= function.normalize_input(input_value)
newInput=input_value[0]
newInput=np.append(newInput,1)
dim_input=np.size(newInput)
x=np.zeros(dim_input)


for i in range(np.size(struct_layers)):
    #inizializza i nostri layer e li stampo a schermo
    struct_layers[i]=Layer.Layer(x,nj[i],nj[i+1],[np.size(x),nj[i]])
    x=np.zeros(nj[i]+1)
#out=sigmoidal.output_nn(struct_layers,input_value[0])
#print("output expected :", output_expected)
#print("expexted loss:",np.square(out-output_expected))   
print("--------------------------------------")
Bp.backprogation(struct_layers, num_epoch, learning_rate, newInput, output_expected)
