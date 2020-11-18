import numpy as np
import matplotlib.pyplot as plt
import csv
import leggifile
import normalized_initialization
import sigmoidal
import Layer

layer=3
learning_rate=0.1
output_expected=[4,8,3,2]
num_epoch=4
filename = "dati.csv"
input_value=leggifile.leggi(filename)
#creo la nuova struttura che conterrà i layer
struct_layers = np.empty(layer,Layer.Layer)
dim_input=np.size(input_value[0])
nj=4 #np.random.randint(1,5)
nj_next=4 #np.random.randint(1,5)
x=np.zeros(dim_input)

for i in range(np.size(struct_layers)):
    
    #inizializza i nostri layer e li stampo a schermo
    struct_layers[i]=Layer.Layer(x,nj,nj_next,[np.size(x),nj])
    x=np.zeros(nj)
    nj=nj_next
    #nj_next=np.random.randint(1,5)

out=sigmoidal.output_nn(struct_layers,input_value[0])
#print("output expected :", output_expected)
#print("expexted loss:",np.square(out-output_expected))   
print("--------------------------------------")

sigmoidal.backprogation(struct_layers,num_epoch,learning_rate,out,output_expected)
