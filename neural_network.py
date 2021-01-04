import numpy as np
import Layer
import matplotlib.pyplot as plt
from function import output_nn, der_loss, derivate_sigmoid, derivate_sigmoid_2, input_matrix, output_matrix, MSE
from ThreadPool import ThreadPool
import graphycs
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
        #self.last_layer = np.size(self.struct_layers) - 1
        #self.nodes_output_layer = self.struct_layers[self.last_layer].nj


    def trainig(self, training_set, validation_set, batch_size, epochs):
        #inizializzo struct_layer
        for i in range(1,self.numero_layer+1):
            self.struct_layers[i-1]=Layer.Layer(self.nj[i],self.nj[i-1],self.nj[i+1],batch_size)
        
        training_set_output = output_matrix(training_set)
        training_set_input = input_matrix(training_set) 
        output_NN = np.zeros(training_set_output.shape)

        validation_set_input = input_matrix(validation_set)
        validation_set_output = output_matrix(validation_set)

        best_w_validation=[self.struct_layers,10000]
        min_error_VL = 10000000000000
        #punti per grafico 
        epo = []
        lo = []
        errors_validation = []
        for i in range(epochs):
            index_matrix = 0#np.random.randint(0, (training_set_input.shape[0] - batch_size)+1 )
            ThreadPool(self.struct_layers, training_set_input, index_matrix, batch_size, output_NN)
            loss = self.backprogation(index_matrix, output_NN, training_set_output, batch_size)
            epo.append(i)
            lo.append(loss)
            if (i % 5 == 0):
                if self.validation(min_error_VL, errors_validation, validation_set_input, validation_set_output):
                    best_w_validation = [self.struct_layers, min_error_VL]
        #graphycs.grafico(epo, lo, "epochs", "loss")
        ThreadPool(self.struct_layers, training_set_input, index_matrix, batch_size, output_NN)
        print("output ", output_NN)
       
       
        #graphycs.grafico(epo, min_error_VL, "epochs", "validation_error")
        output_NN = np.zeros(validation_set_output.shape)
        ThreadPool(best_w_validation[0], validation_set_input, 0, validation_set_input.shape[0], output_NN)
        print("best model ", output_NN)
        #print("accuratezza: \n" ,np.abs((self.output_expected -output_NN) / self.output_expected)*100)
        return best_w_validation

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
    def validation(self,min_error_VL, errors_validation, validation_set_input, validation_set_output):
        output_NN = np.zeros(validation_set_output.shape)
        ThreadPool(self.struct_layers, validation_set_input, 0, validation_set_input.shape[0], output_NN)
        loss_validation = MSE(output_NN, validation_set_output, validation_set_output.shape[0])
        errors_validation.append(loss_validation)
        if loss_validation < min_error_VL:
            min_error_VL = loss_validation
            return True
        return False