import numpy as np
import csv
import Matrix_io
import pandas


#####################
## READING METHODS ##
#####################


def read_csv(type_problem,file_name):
    if (type_problem == "Regression"):   
        file_csv = pandas.read_csv(file_name, delimiter = ',', 
                                    names = ["data", "x_1", "x_2","x_3", "x_4","x_5", "x_6","x_7", "x_8","x_9", "x_10", "label_1", "label_2"], 
                                    skiprows= 7)

        file_csv = file_csv.drop(["data"], axis = 1)
        
        matrix = file_csv.to_numpy()

    elif (type_problem == "classification"):
        
        file_csv = pandas.read_csv(file_name, delimiter = ' ', names = ["label", "x_1", "x_2","x_3", "x_4","x_5", "x_6", "data"])
        
        file_csv = file_csv.reindex(columns=[ "x_1", "x_2","x_3", "x_4","x_5", "x_6","label"])
        
        matrix = file_csv.to_numpy()

    #aggiungi altri modi per leggere diversi file qui sotto

    return matrix


#####################
## WRITING METHODS ##
#####################


def print_result(out_file,stampa):
    print(stampa)
    stampa=stampa+"\n"
    out_file.write(stampa)
    out_file.flush()



    
#separo gli esempi di training
def divide_exaples_hold_out(matrix_input, columns_output):
    #divido 70% TR, 20% VL, 10% TS
    rows = matrix_input.shape[0]
    training_size = rows *70 //100
    validation_size = rows *20 //100
    training = Matrix_io.Matrix_io(matrix_input[0:training_size, :], columns_output)
    validation = Matrix_io.Matrix_io(matrix_input[training_size:training_size+validation_size, :], columns_output)
    test = Matrix_io.Matrix_io(matrix_input[training_size+validation_size:, :], columns_output)
    return [training, validation, test]