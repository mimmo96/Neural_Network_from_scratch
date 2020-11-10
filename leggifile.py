import numpy as np
import csv

def leggi(filename):
    input_value=[]

    with open(filename,'r') as f:
        reader = csv.reader(f,delimiter='\t')
        array=[]

        for line in reader:
            array=line
            input_value.append(array)

    i=0
    dim=np.size(input_value)

    while i<dim:
        print("input ", i+1,":", input_value[i])
        i=i+1

    print("la dimensione è:", np.size(input_value))

    

    