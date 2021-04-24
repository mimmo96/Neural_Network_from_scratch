import numpy as np
from function import LOSS
import math
import pandas
import neural_network

class stat_model:
    '''
        class used for saving result of one neuralnetwork and its iperparameters
    '''
    def __init__(self, NN, mse_tr = -1, mse_vl = -1 ,std= -1, mee = -1, number_model = 0):

        self.NN = NN
        self.mse_tr = mse_tr
        self.mse_vl = mse_vl
        self.mee = mee
        self.number_model = number_model
        self.std = std
    
    def write_result(self, file_name):
        row_csv = {
                'Number_Model' : [self.number_model],
                'Units_per_layer' : [self.NN.nj[1:-1]],
                'learning_rate' : [self.NN.learning_rate],
                'lambda' : [self.NN.v_lambda],
                'alfa' : [self.NN.alfa],
                'function_hidden' : [self.NN.function],
                'inizialization_weights' : [self.NN.type_weight],
                'Error_MSE_tr' : [self.mse_tr],
                'Error_MSE_vl' : [self.mse_vl],
                'Error_MEE' : [self.mee],
                'Variance' : [self.std]
            }
        df = pandas.DataFrame(row_csv)
        df.to_csv(file_name, mode='a', header = False, index=False)

    def getNN(self):
        return self.NN

    def greater_loss(self, model):          
        if self.mse_vl > model.mse_vl:
            return True
        return False

    def lesser_accuracy(self, model):          
        if self.mee < model.mee:
            return True
        else:
            self.greater_loss(model)

class ensemble:
    '''
        class used to contains the top 10 NeuralNetwork and some function for compute mean,loss,etc
    '''
    #NN_array=array containing the top 10 Neural networks
    #data = data to test
    def __init__(self, NN_array =[], data = [], limit = 3):
        self.NN_array=NN_array
        self.data=data
        self.limit = limit
    
    #gives me the average of the network outputs on the data 
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

    #loss calculated on the network outputs
    def loss_average(self):
        #calculate the average of the outputs, it gives me a single output vector
        output=self.output_average()
        loss_test = LOSS(output, self.data.output(), penalty_term = 0)
        return loss_test

    def mean_loss(self):
        mean_mee = 0
        mean_mse_vl = 0
        mean_mse_tr = 0
        
        #used to save a best mse on validation set
        best_mse_vl = + math.inf

        for model in self.NN_array:
            if best_mse_vl > model.mse_vl:
                best_mse_vl = model.mse_vl
                best_NN = model
            
            mean_mee += model.mee
            mean_mse_vl += model.mse_vl 
            mean_mse_tr += model.mse_tr 
        
        mean_mse_tr /= np.size(self.NN_array)
        mean_mse_vl /= np.size(self.NN_array)
        mean_mee /= np.size(self.NN_array)
        
        best_NN.mse_vl = mean_mse_vl
        best_NN.mse_tr = mean_mse_tr
        best_NN.mee = mean_mee

        return best_NN

    #add the model on top 10 if it has the best loss
    def k_is_in_top(self, model, type_problem):
    
        if len(self.NN_array) < self.limit:
            self.NN_array.append(model)
        else:
            self.NN_array = sorted(self.NN_array, key=lambda x : x.mse_vl)
            worst_NN = self.NN_array[-1]
            if (((type_problem == "classification") and worst_NN.lesser_accuracy(model)) or ((type_problem == "Regression") and worst_NN.greater_loss(model))):
                self.NN_array.pop()
                self.NN_array.append(model)
        
    #write result of parameters on external file with csv format
    def write_result(self, file_csv):
        df = pandas.DataFrame(columns = ['Number_Model' ,
                'Units_per_layer',
                'learning_rate' ,
                'lambda' ,
                'alfa' ,
                'function_hidden',
                'inizialization_weights',
                'Error_MSE_tr' ,
                'Error_MSE_vl' ,
                'Error_MEE' ,
                'Variance' ])
        df.to_csv(file_csv)
        for model in self.NN_array:
            model.write_result(file_csv)

    #return the model with the best mee      
    def print_top(self):
        return self.NN_array[0].mee
