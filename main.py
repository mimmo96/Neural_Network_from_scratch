import numpy as np
import leggifile
import function
from Model_selection import model_selection
from random import randint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import Matrix_io

#----------------------------PARAMETRI DA ANALIZZARE----------------------

num_epoch=[400]
filename = "CUP\ML-CUP20-TR.csv"
file_name_test = "CUP\ML-CUP20-TR.csv"
dim_output = 2
problem_type="Regression"
#mi crea i layer in questo modo: (num_input, num_units_layer1, num_units_layer_2, .... , num_output, 0)
#nj=[ [10, 20, 1, 0], [10, 10, 2 , 0], [10, 30, 2 , 0],[10, 100, 2 , 0],[10, 50, 2 , 0],[10, 7, 2 , 0] ]

learning_rate = [0.0021435]
alfa = [0.5]
v_lambda = [0.001]
fun = ["tanh"]
#fun_out non sarà considerata in caso di regressione
fun_out=["Regression"]
weight=["random"]
#-------------------------------FINE PARAMETRI------------------------------

#leggo i dati dal file e li salvo in una matrice

training_set = leggifile.leggi(problem_type,filename)
validation_set = leggifile.leggi(problem_type,filename)
test_set = validation_set

#se sono nella classificazione utilizzo one-hot-coding
if(problem_type=="classification"):
    one = OneHotEncoder(sparse=False)
    label=LabelEncoder()

    training_set_X = one.fit_transform(training_set[:,:-1])
    tmp = np.append(training_set_X,np.zeros([len(training_set_X),1]),1)
    tmp[:, -1] = label.fit_transform(training_set[:, -1])
    training_set = tmp

    validation_set_X = one.fit_transform(validation_set[:,:-1])
    tmp = np.append(validation_set_X,np.zeros([len(validation_set_X),1]),1)
    tmp[:, -1] = label.fit_transform(validation_set[:, -1])
    validation_set = tmp

    test_set_X = one.fit_transform(test_set[:,:-1])
    tmp = np.append(test_set_X,np.zeros([len(test_set_X),1]),1)
    tmp[:, -1] = label.fit_transform(test_set[:, -1])
    test_set = tmp

training_set = function.normalize_input(training_set,dim_output)

newInput=training_set[0]
dim_input=np.size(newInput) - dim_output

#cambio la divisione dei dati a seconda de lmodello che mi ritrovo ad affrontare
if(problem_type=="classification"):
    training_set = Matrix_io.Matrix_io(training_set, dim_output)
    validation_set = Matrix_io.Matrix_io(validation_set, dim_output)
    test_set = Matrix_io.Matrix_io(test_set, dim_output)

else:
    training_set, validation_set, test_set = leggifile.divide_exaples(training_set, dim_output)

nj=[[dim_input,20,20,dim_output,0]]
batch_array=[32,512]

#prima di fare la model selection controllo se il batch è di dimensione giusta
for i in batch_array:
    if(i>training_set.input().shape[0]):
        print("batch troppo grande!!!\nSTOPPO")
        exit()

model_selection(alfa, learning_rate, v_lambda, nj, training_set, validation_set,test_set, batch_array, num_epoch,fun, fun_out, weight,problem_type)

