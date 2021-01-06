import numpy as np
import Layer
import matplotlib.pyplot as plt
from function import der_loss, derivate_sigmoid, derivate_sigmoid_2, MSE
import graphycs
from concurrent.futures import ThreadPoolExecutor
class neural_network:
    
    
    def __init__(self, nj, alfa, v_lambda, learning_rate, numero_layer):

        self.alfa = alfa
        self.v_lambda = v_lambda
        self.learning_rate=learning_rate
        self.nj=nj
        #creo la nuova struttura che conterrà i layer
        #CANCELLARE STRUCT_
        self.struct_layers = np.empty(numero_layer, Layer.Layer)
        self.numero_layer = numero_layer
        for i in range(1,self.numero_layer+1):
            self.struct_layers[i-1]=Layer.Layer(self.nj[i],self.nj[i-1],0.7)
        #self.last_layer = np.size(self.struct_layers) - 1
        #self.nodes_output_layer = self.struct_layers[self.last_layer].nj

    #FORWARDING
    def forward (self, x_input, row_input_layer, validation = False):
        i = np.size(self.struct_layers)-1
        for layer in self.struct_layers:
            x_input = np.append(x_input, 1)
            if validation == False:
                layer.x[row_input_layer,:] = x_input  
            #hidden layer
            if i != 0:
                x_input = layer.output(x_input)
            #output layer
            else:
                output=np.zeros(layer.nj)
                for nj in range(layer.nj):
                    output[nj]=layer.net(nj, x_input)
            i = i - 1
        return output

    def task_forwarding(self, x_input, i, output, row_input_layer, validation = False):
        output[i, :] = self.forward(x_input, row_input_layer, validation)

    def ThreadPool_Forward(self, matrix_input, index_matrix, batch_size, output, validation = False):
        executor = ThreadPoolExecutor(10)
        max_i = batch_size + index_matrix
        for i in range(index_matrix, max_i):
            row_input_layer = i % batch_size
            executor.submit(self.task_forwarding, matrix_input[i, :], i, output, row_input_layer, validation)
        executor.shutdown(True)

        return output
    
    #TRAINING
    def trainig(self, training_set, validation_set, batch_size, epochs):
        #inizializzo struct_layer
        for layer in self.struct_layers:
            layer.set_x(batch_size)

        training_set_output = training_set.output()
        training_set_input = training_set.input() 
        output_NN = np.zeros(training_set_output.shape)

        best_min_err_validation = -1
        #punti per grafico 
        epo = []
        lo = []
        errors_validation = []
        for i in range(epochs):
            index_matrix = np.random.randint(0, (training_set_input.shape[0] - batch_size)+1 )
            self.ThreadPool_Forward(training_set_input, index_matrix, batch_size, output_NN)
            loss = self.backprogation(index_matrix, output_NN, training_set_output, batch_size)
            epo.append(i)
            lo.append(loss)
            if (i % 5 == 0):
                best_min_err_validation = self.validation(best_min_err_validation, errors_validation, validation_set)
        #graphycs.grafico(epo, lo, "epochs", "loss")
        #ThreadPool_Forward(self.struct_layers, training_set_input, index_matrix, batch_size, output_NN)
        print("output ", output_NN)
    
        #graphycs.grafico(epo, best_min_err_validation, "epochs", "validation_error")
        output_NN = np.zeros(validation_set.output().shape)
        self.ThreadPool_Forward(validation_set.input(), 0, validation_set.input().shape[0], output_NN, True)
        print("validation len ", validation_set.output().shape[0], "error_best_model", best_min_err_validation, "best model ", output_NN)
        #print("accuratezza: \n" ,np.abs((self.output_expected -output_NN) / self.output_expected)*100)
        return best_min_err_validation

    def backprogation(self, index_matrix, output_NN, training_set_output, batch_size):
        
        for i in range(np.size(self.struct_layers) - 1, -1, -1):
            # restituisce l'oggetto layer i-esimo
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
                gradient = np.dot(delta[j,:],layer.x) - self.v_lambda*layer.w_matrix[:, j]*2
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
    def validation(self,best_min_err_validation, errors_validation, validation_set):
        validation_set_input = validation_set.input()
        validation_set_output = validation_set.output()
        output_NN = np.zeros(validation_set_output.shape)
        self.ThreadPool_Forward(validation_set_input, 0, validation_set_input.shape[0], output_NN, True)
        loss_validation = MSE(output_NN, validation_set_output, validation_set_output.shape[0])
        errors_validation.append(loss_validation)
        if (loss_validation < best_min_err_validation) | (best_min_err_validation == -1):
            best_min_err_validation = loss_validation
            print(best_min_err_validation)
        return best_min_err_validation