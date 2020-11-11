import numpy as np
import matplotlib.pyplot as plt 
import csv
import leggifile
import normalized_initialization
import sigmoidal

layer=5
filename = "dati.csv"
input_value=leggifile.leggi(filename)

w=normalized_initialization.init_w(2,4,[np.size(input_value[0]),layer])
net=np.dot(input_value[0],w)

plt.plot(net, sigmoidal.sigmoid(net)) 
plt.xlabel("x") 
plt.ylabel("Sigmoid(X)") 
    
plt.show() 
  

  


    