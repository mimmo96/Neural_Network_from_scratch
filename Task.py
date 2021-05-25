import numpy as np
import Read_write_file
import Function
from Hold_out import Hold_out
import Matrix_io
import itertools
from CV_k_fold import cv_k_fold
import pandas
import File_names as fn

class Regression:
    '''
        class used for regression problem
    '''
    def __init__(self, file_data, num_epoch, dim_output, hidden_units, batch_array, learning_rate_init, type_learning_rate, alfa, v_lambda,
                        fun, weight, early_stopping,num_training):

        self.type_problem="Regression"
        self.num_epoch = num_epoch
        self.dim_output = dim_output
        self.data_set = Read_write_file.read_csv(self.type_problem,file_data)
        self.test_set = [[]]
        #self.data_set = Function.normalize_input(self.data_set,self.dim_output)
      
        #HYPERPARAMETERS
        self.hidden_units=hidden_units
        self.batch_array=batch_array
        self.learning_rate_init = learning_rate_init
        self.type_learning_rate = type_learning_rate
        self.alfa = alfa
        self.v_lambda = v_lambda
        self.fun = fun      
        self.fun_out=["linear"]
        self.weight=weight
        self.early_stopping = early_stopping
        self.top_models = []
        self.grid = list(itertools.product(self.early_stopping, self.batch_array, self.type_learning_rate, self.fun, self.fun_out, self.hidden_units, self.learning_rate_init, self.alfa, self.v_lambda,self.weight))
        self.dimension_grid = len(self.grid)
        self.num_training=num_training

    ############
    # HOLD OUT #
    ############

    def startexecution_Hold_out(self,):
        training_set, validation_set, self.test_set = Function.divide_exaples_hold_out(self.data_set, self.dim_output)
        self.dim_input = training_set.get_len_input()
        
        for i in self.batch_array:
            if(i > training_set.input().shape[0]):
                print("batch.size out of bounded\n")
                exit()

        self.top_models,self.num_training  = Hold_out(self.num_epoch, self.grid,training_set, validation_set, self.test_set, self.type_problem, self.dimension_grid,self.num_training)
        return self.top_models
    
    #############
    # CV-K-FOLD #
    #############
    
    def startexecution_k_fold(self,):
        self.devolopment_set, self.test_set = Function.divide_exaples_k_fold(self.data_set, self.dim_output)
        self.dim_input = self.devolopment_set.get_len_input()
        
        for i in self.batch_array:
            if(i > self.devolopment_set.input().shape[0]):
                print("batch.size out of bounded\n")
                exit()

        self.top_models,self.num_training = cv_k_fold(self.num_epoch, self.grid, self.devolopment_set, self.test_set, self.type_problem,self.num_training)
        return self.top_models
    
    ##############
    # BLIND TEST #
    ##############
    
    def blind_test(self, file_name):
        blind_set = Read_write_file.read_csv("blind_test", file_name)
        output = np.zeros((blind_set.shape[0], self.dim_output ))
        for model in self.top_models:
            output_NN = np.zeros((blind_set.shape[0], self.dim_output))
            model.NN.Forwarding(blind_set, output_NN, True)
            output += output_NN
        output = np.divide(output,np.size(self.top_models))

        pandas.DataFrame(output).to_csv(fn.result_blind_test, mode='a', header = True)
    
    def retraining (self, models):
        mee_results = []
        for model in models:
            mee_results.append(model.NN.retraining(self.devolopment_set))
        return models, mee_results

class Classification:
    '''
        class used for classification problem
    '''
    def __init__(self,file_data_tr, file_data_ts, num_epoch,dim_output,hidden_units,batch_array,learning_rate_init, learning_rate_type, alfa,v_lambda,
                        fun,fun_out,weight,early_stopping):

        self.type_problem="classification"
        self.num_epoch = num_epoch
        self.dim_output = dim_output
        self.training_set = Read_write_file.read_csv(self.type_problem,file_data_tr)
        self.validation_set = Read_write_file.read_csv(self.type_problem,file_data_ts)
        self.test_set = self.validation_set
        
        self.training_set = Function.one_hot_encoding(self.training_set)
        self.validation_set = Function.one_hot_encoding(self.validation_set)
        self.test_set = Function.one_hot_encoding(self.test_set)
        
        self.training_set = Matrix_io.Matrix_io(self.training_set, dim_output)
        self.validation_set = Matrix_io.Matrix_io(self.validation_set, dim_output)
        self.test_set = Matrix_io.Matrix_io(self.test_set, dim_output)
        self.dim_input = self.training_set.get_len_input()
        
        ###################
        # HYPERPARAMETERS #
        ###################
        
        self.hidden_units=hidden_units
        self.batch_array=batch_array
        self.learning_rate_init = learning_rate_init
        self.learning_rate_type = learning_rate_type
        self.alfa = alfa
        self.v_lambda = v_lambda
        self.fun = fun      
        self.fun_out=fun_out
        self.weight=weight
        self.early_stopping = early_stopping

        self.grid = list(itertools.product(self.early_stopping, self.batch_array, self.learning_rate_type, self.fun, self.fun_out, self.hidden_units, self.learning_rate_init, self.alfa, self.v_lambda,self.weight))  
        self.dimension_grid = len(self.grid)


    def startexecution(self,):

        for i in self.batch_array:
            if(i>self.training_set.input().shape[0]):
                print("batch.size out of bounded\n")
                exit()

        self.top_models = Hold_out(self.num_epoch, self.grid, self.training_set, self.validation_set, self.test_set, self.type_problem, self.dimension_grid)
        return self.top_models