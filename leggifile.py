import numpy as np
import csv
import argparse
import Matrix_io
parser = argparse.ArgumentParser(description='Process some integers.')

def leggi(problem_type,filename):

    with open(filename, newline='') as csvfile:
        data = list(csv.reader(csvfile))

    #file di classificazione
    if(problem_type=="classification"):
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
            temp=stringa[0]
    
            for k in range (0,np.size(stringa)-1):
                if(k==np.size(stringa)-2):
                    stringa[k]=temp
                else:
                    stringa[k]=float(stringa[k+1])
    
            for j in range(colonne):
                    matriceinput[i,j] = stringa[j]

        return matriceinput
    
    #file di regressione
    else:
        #numero di valori in input
        righe = len(data)
        colonne = 12
        #array che conterra [#input][#array di valori interi]
        matriceinput=np.zeros([righe-7, colonne])

        #scandisco le righe e salvo i valori in un array
        #parto da 7 come indice perchè escludo i commenti iniziali
        for i in range(7,righe):
            array = np.array(data[i], dtype="float")
    
            stringa=data[i]
    
            #trasformo in float  
            cazzo=np.zeros(np.size(stringa)-1)
            for k in range (0,np.size(stringa)-1):
                cazzo[k]=float(stringa[k+1])

            for j in range(colonne):
                matriceinput[i-7,j] = cazzo[j]

        return matriceinput

#separo gli esempi di training
def divide_exaples(matrix_input, columns_output):
    #divido 70% TR, 20% VL, 10% TS
    rows = matrix_input.shape[0]
    training_size = rows *70 //100
    validation_size = rows *20 //100
    training = Matrix_io.Matrix_io(matrix_input[0:training_size, :], columns_output)
    validation = Matrix_io.Matrix_io(matrix_input[training_size:training_size+validation_size, :], columns_output)
    test = Matrix_io.Matrix_io(matrix_input[training_size+validation_size:, :], columns_output)
    return [training, validation, test]

def print_result(out_file,stampa):
    print(stampa)
    stampa=stampa+"\n"
    out_file.write(stampa)