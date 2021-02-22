import numpy as np
import csv
import argparse
import Matrix_io
parser = argparse.ArgumentParser(description='Process some integers.')

def leggi(filename):

    with open(filename, newline='') as csvfile:
        data = list(csv.reader(csvfile))

    #numero di valori in input
    righe = len(data)
    colonne = 7
    #array che conterra [#input][#array di valori interi]
    matriceinput=np.zeros([righe, colonne])

    #scandisco le righe e salvo i valori in un array
    for i in range(righe):
        stringa=data[i][0]
        stringa=stringa.split()
        stringa[np.size(stringa)-1]=0
     
        for k in range (0,np.size(stringa)-1):
            stringa[k]=float(stringa[k])
     
        for j in range(colonne):
                matriceinput[i,j] = stringa[j]

    return matriceinput

def divide_exaples(matrix_input, columns_output):
    rows = matrix_input.shape[0]
    training_size = rows // 2
    validation_size = (rows - training_size) // 2

    training = Matrix_io.Matrix_io(matrix_input[0:training_size, :], columns_output)
    validation = Matrix_io.Matrix_io(matrix_input[training_size:training_size+validation_size, :], columns_output)
    test = Matrix_io.Matrix_io(matrix_input[training_size+validation_size:, :], columns_output)
    return [training, validation, test]