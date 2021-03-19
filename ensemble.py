import numpy as np
from function import LOSS
import math
import pandas
import neural_network
class stat_model:
    def __init__(self, NN, mse = -1, mee = -1, number_model = 0):

        self.NN = NN
        self.mse = mse
        self.mee = mee
        self.number_model = number_model
    
    

class ensemble:
    
    #NN_array=array contenente le migliroi 10 Neural network
    #data= dati su cui testare
    def __init__(self, NN_array =[], data = [], limit = 1):
        self.NN_array=NN_array
        self.data=data
        self.limit = limit
    
    #mi restituisce la media degli output della rete sui dati
    def output_average(self):
        output=0
        count=0
        for NN in self.NN_array.NN:
            output_NN = np.zeros(self.data.output().shape)
            NN.ThreadPool_Forward(self.data.input(), 0, self.data.input().shape[0], output_NN, True)
            output=output+output_NN
            count=count+1
        output=np.divide(output,count)
        return output

    #loss carcolata sugli output della rete
    def loss_average(self):
        #calcolo la media degli output, mi restituisce un unico vettore di output
        output=self.output_average()
        loss_test = LOSS(output, self.data.output(), self.data.output().shape[0], penalty_term = 0)
        return loss_test

    def mean_loss(self):
        #AGGIUNGERE DEVIAZIONE STANDARD
        mean_mee = 0
        mean_mse = 0
        best_mse = + math.inf
        best_mee = + math.inf

        for model in self.NN_array:
            if best_mee > model.mee:
                best_mee = model.mee
                best_mse = model.mse
                best_NN = model
            mean_mee += model.mee
            mean_mse += model.mse 
        
        mean_mse /= np.size(self.NN_array)
        mean_mee /= np.size(self.NN_array)
        best_NN.mse = mean_mse
        best_NN.mee = mean_mee

        return best_NN

    def k_is_in_top(self, model):
    
        if len(self.NN_array) < self.limit:
            self.NN_array.append(model)
        else:
            NN_array = sorted(self.NN_array, key=lambda x : x.mee)
            worst_NN = NN_array[-1]
            if worst_NN.mee > model.mee:
                NN_array.pop()
                self.NN_array.append(model)
        
    
    def write_result(self, file_csv):
        df = pandas.DataFrame(columns = ['Number_Model' ,
                'Units_per_layer',
                'learning_rate' ,
                'lambda' ,
                'alfa' ,
                'function_hidden',
                'inizialization_weights',
                'Error_MSE' ,
                'Error_MEE' ])
        for model in self.NN_array:
            NN = model.NN
            mse = model.mse
            mee = model.mee
            number_model = model.number_model
            row_csv = {
                'Number_Model' : [number_model],
                'Units_per_layer' : [NN.nj[1:-1]],
                'learning_rate' : [NN.learning_rate],
                'lambda' : [NN.v_lambda],
                'alfa' : [NN.alfa],
                'function_hidden' : [NN.function],
                'inizialization_weights' : [NN.type_weight],
                'Error_MSE' : [mse],
                'Error_MEE' : [mee]
            }
            df = pandas.DataFrame(row_csv)
            df.to_csv(file_csv, mode='a')
            

    def print_top(self):
        return self.NN_array[0].mee