from itertools import count
from os import write

from numpy.lib.function_base import insert
from Function import MEE
import numpy as np
from Function import LOSS
import math
import pandas

class Stat_model:
    '''
        class used for saving result of one neuralnetwork and its iperparameters
    '''
    def __init__(self, NN, MSE_tr = -1, MSE_vl = -1 , MEE_tr = -1, MEE_vl = -1, accuracy_tr = -1, accuracy_vl = -1, std= -1, number_model = 0, MSE_ts=-1 ,MEE_ts=-1):

        self.NN = NN
        self.MSE_tr = MSE_tr
        self.MSE_vl = MSE_vl
        self.MEE_tr = MEE_tr
        self.MEE_vl = MEE_vl    
        self.accuracy_tr = accuracy_tr
        self.accuracy_vl = accuracy_vl
        self.number_model = number_model
        self.std = std
        self.MSE_ts= MSE_ts
        self.MEE_ts= MEE_ts
    
    def write_result(self, file_name):
        row_csv = {
                'Number_Model' : [self.number_model],
                'Units_per_layer' : [self.NN.units[1:-1]],
                'learning_rate' : [self.NN.learning_rate],
                'lambda' : [self.NN.v_lambda],
                'alfa' : [self.NN.alfa],
                'function_hidden' : [self.NN.function],
                'inizialization_weights' : [self.NN.type_weight],
                'batch_size' : [self.NN.batch_size],
                'Error_MSE_tr' : [self.MSE_tr],
                'Error_MSE_vl' : [self.MSE_vl],
                'Error_MEE_tr' : [self.MEE_tr],
                'Error_MEE_vl' : [self.MEE_vl],
                'Accuracy_tr' : [self.accuracy_tr],
                'Accuracy_vl' : [self.accuracy_vl],
                'Variance' : [self.std]
            }
        df = pandas.DataFrame(row_csv)
        df.to_csv(file_name, mode='a', header = False, index=False)

    def write_cost(self,file_name):
        row_csv = {
                'Number_Model' : [self.number_model],
                'Error_MSE_ts' : [self.MSE_ts],
                'Error_MEE_ts' : [self.MEE_ts]
        }
        
        df = pandas.DataFrame(row_csv)
        df.to_csv(file_name, mode='a', header = False, index=False)

    def getNN(self):
        return self.NN

    def greater_loss(self, model):          
        if self.MSE_vl > model.MSE_vl:
            return True
        return False

    def lesser_accuracy(self, model):          
        if self.accuracy_vl < model.accuracy_vl:
            return True
        else:
            self.greater_loss(model)

class Ensemble:
    '''
        class used to contains the top 10 NeuralNetwork and some function for compute mean,loss,etc
    '''
    #NN_array=array containing the top 10 Neural networks of type stat_model
    #data = data to test
    def __init__(self, NN_array =[], limit = 8, type_problem = "Regression"):
        self.NN_array=[]
        self.limit = limit
        self.type_problem = type_problem
        for elm in NN_array:
            self.insert_model(elm)
    
    #gives me the average of the network outputs on the data and save the loss/mee
    def output_average(self, data_set,file_name):
        output = np.zeros(data_set.output().shape)
        count = 0
        for model in self.NN_array:
            output_NN = np.zeros(data_set.output().shape)
            model.NN.Forwarding(data_set.input(), output_NN, True)
            model.MSE_ts = LOSS(output_NN, data_set.output())
            model.MEE_ts= MEE(output_NN, data_set.output())
            model.write_cost(file_name)
            output += output_NN
            count += 1
        output = np.divide(output,count)
        mee=MEE(output, data_set.output())
        mse=LOSS(output, data_set.output())
        return output,mse,mee

    #loss calculated on the network outputs
    def loss_average(self, data_set,file_name):
        #calculate the average of the outputs, it gives me a single output vector
        output,mee,mse = self.output_average(data_set,file_name)
        row_csv = {
                'Error_MSE_ts' : [mse],
                'Error_MEE_ts' : [mee]
        }
        pandas.DataFrame(row_csv).to_csv(file_name, mode='a', header = False, index=False)
        return mse, mee
    

    #return best model and the mean
    def best_neural_network(self):
        mean_MEE_tr = 0
        mean_MEE_vl = 0
        mean_MSE_vl = 0
        mean_MSE_tr = 0
        mean_accuracy_tr = 0
        mean_accuracy_vl = 0
        MSE_vl_array=[]
        #used to save a best mse on validation set
        best_MSE_vl = + math.inf

        for model in self.NN_array:
            if best_MSE_vl > model.MSE_vl:
                best_MSE_vl = model.MSE_vl
                best_NN = model
            
            mean_MEE_tr += model.MEE_tr
            mean_MEE_vl += model.MEE_vl
            mean_MSE_vl += model.MSE_vl 
            mean_MSE_tr += model.MSE_tr 
            mean_accuracy_tr += model.accuracy_tr
            mean_accuracy_vl += model.accuracy_vl
            MSE_vl_array.append(model.MSE_vl)

        mean_MSE_tr /= np.size(self.NN_array)
        mean_MSE_vl /= np.size(self.NN_array)
        mean_MEE_tr /= np.size(self.NN_array)
        mean_MEE_vl /= np.size(self.NN_array)
        mean_accuracy_tr /= np.size(self.NN_array)
        mean_accuracy_vl /= np.size(self.NN_array)

        best_NN.MSE_vl = mean_MSE_vl
        best_NN.MSE_tr = mean_MSE_tr
        best_NN.accuracy_vl = mean_accuracy_vl
        best_NN.accuracy_tr = mean_accuracy_tr
        best_NN.MEE_tr = mean_MEE_tr
        best_NN.MEE_vl = mean_MEE_vl
        best_NN.std = np.std(MSE_vl_array)

        return best_NN

    #add the model on top 10 if it has the best loss
    def insert_model(self, model):
    
        if len(self.NN_array) < self.limit:
            self.NN_array.append(model)
        else:
            self.NN_array = sorted(self.NN_array, key=lambda x : x.MSE_vl)
            worst_NN = self.NN_array[-1]
            if (((self.type_problem == "classification") and worst_NN.lesser_accuracy(model)) or ((self.type_problem == "Regression") and worst_NN.greater_loss(model))):
                self.NN_array.pop()
                self.NN_array.append(model)
        
    #write result of parameters on external file with csv format
    def write_result(self, file_csv):
        for model in self.NN_array:
            model.write_result(file_csv)
    

    #return the model with the best mee      
    def print_top(self):
        return self.NN_array[0].MEE_vl
    
    def getNN(self):
        return self.NN_array
