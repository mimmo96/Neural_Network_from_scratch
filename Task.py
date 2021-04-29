import numpy as np
import Read_write_file
import Function
from Hold_out import Hold_out
import Matrix_io
import itertools
import pandas
import File_names as fn

class Regression:
    '''
        class used for regression problem
    '''
    def __init__(self,num_epoch,dim_output,hidden_units,batch_array,learning_rate,alfa,v_lambda,
                        fun,fun_out,weight,early_stopping):

        self.type_problem="Regression"
        self.num_epoch = num_epoch
        self.dim_output = dim_output
        self.training_set = Read_write_file.read_csv(self.type_problem,fn.ML_cup)
        self.blind_set = Read_write_file.read_csv("blind_test",fn.blind_test)
        self.training_set = Function.normalize_input( self.training_set,self.dim_output)
        self.training_set, self.validation_set, self.test_set = Function.divide_exaples_hold_out( self.training_set, self.dim_output)
        self.dim_input = self. training_set.get_len_input()
        
        ###################
        # HYPERPARAMETERS #
        ###################
        
        self.hidden_units=hidden_units
        self.batch_array=batch_array
        self.learning_rate = learning_rate
        self.alfa = alfa
        self.v_lambda = v_lambda
        self.fun = fun      
        self.fun_out=["Regression"]
        self.weight=weight
        self.early_stopping = early_stopping

        self.grid = list(itertools.product(self.early_stopping, self.batch_array, self.fun, self.fun_out, self.hidden_units, self.learning_rate, self.alfa, self.v_lambda,self.weight))

    def startexecution(self,):
        
        for i in self.batch_array:
            if(i>self.training_set.input().shape[0]):
                print("batch.size out of bounded\n")
                exit()

        ############
        # HOLD-OUT #
        ############
        top_models = Hold_out(self.num_epoch,self.grid,self.training_set, self.validation_set, self.test_set, self.type_problem)

        ##############
        # BLIND TEST #
        ##############
    
        output = np.zeros(( self.blind_set.shape[0], self.training_set.output().shape[1]))

        for model in top_models:
            output_NN = np.zeros(( self.blind_set.shape[0], self.training_set.output().shape[1]))
            model.NN.Forwarding( self.blind_set, output_NN, True)
            output += output_NN
        output = np.divide(output,np.size(top_models))

        pandas.DataFrame(output).to_csv(fn.result_blind_test, mode='a', header = False)
        

class Classification:
    '''
        class used for regression problem
    '''
    def __init__(self,num_epoch,dim_output,hidden_units,batch_array,learning_rate,alfa,v_lambda,
                        fun,fun_out,weight,early_stopping):

        self.type_problem="classification"
        self.num_epoch = num_epoch
        self.dim_output = dim_output
        self.training_set = Read_write_file.read_csv(self.type_problem,fn.Monk_1_tr)
        self.validation_set = Read_write_file.read_csv(self.type_problem,fn.Monk_1_ts)
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
        self.learning_rate = learning_rate
        self.alfa = alfa
        self.v_lambda = v_lambda
        self.fun = fun      
        self.fun_out=fun_out
        self.weight=weight
        self.early_stopping = early_stopping

        self.grid = list(itertools.product(self.early_stopping, self.batch_array, self.fun, self.fun_out, self.hidden_units, self.learning_rate, self.alfa, self.v_lambda,self.weight))


    def startexecution(self,):

        for i in self.batch_array:
            if(i>self.training_set.input().shape[0]):
                print("batch.size out of bounded\n")
                exit()

        top_models = Hold_out(self.num_epoch,self.grid,self.training_set, self.validation_set, self.test_set, self.type_problem)
