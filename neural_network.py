import numpy as np
import Layer
import matplotlib.pyplot as plt
from function import der_loss, derivate_sigmoid, derivate_sigmoid_2, MSE
import graphycs
from concurrent.futures import ThreadPoolExecutor
import Matrix_io

class neural_network:
    
    def __init__(self, nj, alfa, v_lambda, learning_rate, numero_layer):

        self.alfa = alfa
        self.v_lambda = v_lambda
        self.learning_rate=learning_rate
        self.nj=nj
        self.struct_layers = np.empty(numero_layer, Layer.Layer)
        self.numero_layer = numero_layer
       
        for i in range(1,self.numero_layer+1):
            self.struct_layers[i-1]=Layer.Layer(self.nj[i],self.nj[i-1],0.7)

    #   --------METODI CHE ERANO NEI THREAD------------
    
    def forward (self, x_input, row_input_layer, validation = False):
        i = np.size(self.struct_layers)-1
        for layer in self.struct_layers:
            x_input = np.append(x_input, 1)
            if validation == False:
                layer.x[row_input_layer,:] = x_input  
                #print("layer.x[row_input_layer,:]",layer.x[row_input_layer,:])
            #hidden layer
            if i != 0:
                x_input = layer.output(x_input)
            #output layer
            else:
                output=np.zeros(layer.nj)
                for nj in range(layer.nj):
                    output[nj]=layer.net(nj, x_input)
                #print("forward output ", output)
            i = i - 1
        return output

    def task_forwarding(self, x_input, i, output, row_input_layer, validation = False):
        #print("output ", self.forward(x_input, row_input_layer, validation))        
        output[i, :] = self.forward(x_input, row_input_layer, validation)
        #print("Aaaaaaaaaaa:", output[i, :])

    def ThreadPool_Forward(self, matrix_input, index_matrix, batch_size, output, validation = False):
        executor = ThreadPoolExecutor(10)
        max_i = batch_size + index_matrix
        '''
        print("-----------------------------------------")
        print("max_i",max_i)
        print("batch_size",batch_size)
        print("index_matrix",index_matrix)
        print("index_matrix",matrix_input)
        print("output",output)
        
        '''
        for i in range(index_matrix, max_i):
            row_input_layer = i % batch_size
            #print("matrix_input[i, :]",matrix_input[i, :])
            #print("row_input_layer",row_input_layer)
            #print("i",i)
            executor.submit(self.task_forwarding, matrix_input[i, :], i, output, row_input_layer, validation)
        #print("--------------------END-------------------------")
        executor.shutdown(True)

        return output
    
    #   --------FINE METODI CHE ERANO NEI THREAD------------
    
    def trainig(self, training_set, validation_set, batch_size, epochs):
        
        for layer in self.struct_layers:
            layer.set_x(batch_size)

        training_set_input = training_set.input() 
        training_set_output = training_set.output()
        print("----------------------MEDIA OUTPUT TRAINING SET ----------------------------------\n", np.sum(training_set_output) / training_set_output.shape[0])
        best_min_err_validation = -1
        errors_validation = []
        index_matrix = 1
        best_w = 0

        for i in range(epochs):
            
            mini_batch = training_set.create_batch(batch_size)
            for batch in mini_batch:
                output_NN = np.zeros(batch.output().shape)
                self.ThreadPool_Forward(batch.input(),0, batch_size, output_NN)  
                loss = self.backprogation(0, output_NN, batch.output(), batch_size)
                #print("output previsto: \n",batch.output(), "\noutput effettivo: \n", output_NN, )
            if (i % 5 == 0):
                best_min_err_validation = self.validation(best_min_err_validation, best_w, errors_validation, validation_set)

        output_NN = np.zeros(training_set_output.shape)
        self.ThreadPool_Forward(training_set_input, 0, training_set_input.shape[0], output_NN, True)
        print("--------------------------TRAINING RESULT----------------------") 
        print("alfa:",self.alfa, "  lamda:", self.v_lambda, "  learning_rate:",self.learning_rate ,"  nj:",self.nj)
        print("errore medio training: \n" ,np.sum(np.abs(training_set_output - output_NN))/training_set_output.shape[0] )
        print("output previsto: \n",training_set_output, "\noutput effettivo: \n", output_NN )
        print("------------------------------FINE-----------------------------")
        
        '''
        output_NN = np.zeros(validation_set.output().shape)
        self.ThreadPool_Forward(validation_set.input(), validation_set.input().shape[0], output_NN, True)
        print("----------------------------------------------------------------")
        print( "validation_set", validation_set.output(),"\nerror_best_model", best_min_err_validation, "\nbest model ", output_NN)
        print("----------------------------------------------------------------")
        '''
        return best_min_err_validation

    def backprogation(self, index_matrix, output_NN, training_set_output, batch_size):
        
        for i in range(np.size(self.struct_layers) - 1, -1, -1):
            layer = self.struct_layers[i]
            delta = np.empty([layer.nj,batch_size],float)
            der_sig=np.empty(batch_size,float)

            for j in range(0, layer.nj):
                #outputlayer
                if i == (np.size(self.struct_layers) - 1):
                    max_row = index_matrix+batch_size
                    delta[j,:] = der_loss(output_NN[index_matrix:max_row,j], 
                                        training_set_output[index_matrix:max_row,j]) 
                    
                    #calcolo della loss
                    loss = MSE(output_NN[index_matrix:max_row,j],training_set_output[index_matrix:max_row,j], batch_size)
                #hiddenlayer
                else:
                    der_sig = derivate_sigmoid_2(layer.net_matrix(j))
                    #product è un vettore delta*pesi
                    product = delta_out.T.dot(self.struct_layers[i + 1].w_matrix[j, :])
                    for k in range(batch_size):
                        delta[j,k]=np.dot(product[k],der_sig[k])
                #regolarizzazione di thikonov
                gradient = np.dot(delta[j,:],layer.x) - np.dot(self.v_lambda,layer.w_matrix[:, j])*2
                gradient = np.divide(gradient,batch_size)
                Dw_new = np.dot(gradient, self.learning_rate)
                #momentum
                Dw_new = self.DeltaW_new(Dw_new, layer.Delta_w_old[:,j])
                layer.Delta_w_old[:,j] = Dw_new
                #update weights
                layer.w_matrix[:, j] = np.add(layer.w_matrix[:, j], Dw_new)
            delta_out = delta
        return loss

    def DeltaW_new(self,Dw_new,D_w_old):
        return np.add(Dw_new, np.dot(self.alfa, D_w_old))

    #return TRUE if this is the best model
    def validation(self,best_min_err_validation, best_w, errors_validation, validation_set):
        validation_set_input = validation_set.input()
        validation_set_output = validation_set.output()
        output_NN = np.zeros(validation_set_output.shape)
        self.ThreadPool_Forward(validation_set_input, 0, validation_set_input.shape[0], output_NN, True)
        loss_validation = MSE(output_NN, validation_set_output, validation_set_output.shape[0])
        errors_validation.append(loss_validation)
        if (loss_validation < best_min_err_validation) | (best_min_err_validation == -1):
            best_w = self.struct_layers
            best_min_err_validation = loss_validation
        return best_min_err_validation