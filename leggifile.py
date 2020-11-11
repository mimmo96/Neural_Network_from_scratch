import numpy as np
import csv
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')


def leggi(filename):

    with open(filename, newline='') as csvfile:
        data = list(csv.reader(csvfile))

    #numero di valori in input
    dim=len(data)

    #array che conterra [#input][#array di valori interi]
    arrayinput=[]

    #scandisco le righe e salvo i valori in un array
    i=0
    while i<dim:
        array=np.array(data[i],dtype="int")
        arrayinput.append(array)
        i=i+1

    return arrayinput




    

    