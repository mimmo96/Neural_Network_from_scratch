import numpy as np
import csv
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

def leggi(filename):

    with open(filename, newline='') as csvfile:
        data = list(csv.reader(csvfile))

    #numero di valori in input
    righe = len(data)
    colonne = len(data[0])
    #array che conterra [#input][#array di valori interi]
    matriceinput=np.zeros([righe, colonne])

    #scandisco le righe e salvo i valori in un array
    for i in range(righe):
        array = np.array(data[i], dtype="float")
        for j in range(colonne):
                matriceinput[i,j] = array[j]

    return matriceinput
