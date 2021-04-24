from os import fsdecode
import numpy as np
import Layer
import matplotlib
from graphycs import graph
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from function import der_loss, LOSS,  _derivate_activation_function, accuracy, MEE, fun_early_stopping
import backpropagation as bp

class neural_network:
    
    def __init__(self, nj, alfa, v_lambda, learning_rate, num_layers, function, fun_out, type_weight, type_problem = "Regression", early_stopping = False):

        self.alfa = alfa
        self.v_lambda = v_lambda
        
        if (learning_rate >= 1) | (learning_rate == 0):
            self.tau = np.power(2, learning_rate)
        else:
            self.learning_rate=learning_rate
            self.tau = 0
        self.nj=nj
        self.function=function
        self.fun_out = fun_out
        self.struct_layers = np.empty(num_layers, Layer.Layer)
        self.num_layers = num_layers
        self.type_weight=type_weight
        self.type_problem = type_problem
        self.early_stopping = early_stopping
        self.units = []
        for i in range(1,self.num_layers+1):
            self.struct_layers[i-1]=Layer.Layer(self.nj[i],self.nj[i-1],type_weight)
            self.units.append(nj[i])

    ####################
    # FORWARDING PHASE #
    ####################
    def forward (self, x_input, row_input_layer, validation = False):
        i = np.size(self.struct_layers)-1
        for layer in self.struct_layers:
            x_input = np.append(x_input, 1)
            if validation == False:
                layer.x[row_input_layer,:] = x_input  
            
            #hidden layer
            if i != 0:
                #x_input contain the layer's output and input of the next layer 
                x_input = layer.output(self.function, x_input)
            
            #output layer
            else:
                output = np.zeros(layer.nj)
                output = layer.output(self.fun_out, x_input)
                
            i = i - 1
        return output
        
    def Forwarding(self, matrix_input, batch_size, output, validation = False):
        
        for i in range(batch_size):
            row_input_layer = i % batch_size   
            output[i, :] = self.forward(matrix_input[i, :], row_input_layer, validation)
        
        return output
    

    ##################
    # TRAINING PHASE #
    ##################

    def trainig(self, training_set, validation_set, batch_size, epochs,num_training):
        
        for layer in self.struct_layers:
            layer.set_x(batch_size)

        best_loss_validation= + math.inf
        best_model = self.struct_layers
        validation_stop = 3

        loss_array_training = np.empty(0)
        loss_array_validation = np.empty(0)
        accuracy_array_training = np.empty(0)
        accuracy_array_validation = np.empty(0)

        for i in range(epochs):
            #create k mini-batchs of size batch_size
            mini_batch = training_set.create_batch(batch_size)
            
            #if it's using variable learning rate then update it
            if self.tau != 0:
                self.learning_rate = 1 / (1 + (i / self.tau))

            for batch in mini_batch:
                output_NN = np.zeros(batch.output().shape)
                self.Forwarding(batch.input(), batch_size, output_NN)   
                self.backprogation(output_NN, batch.output(), batch_size)

            #loss on all training set 
            output_NN = np.zeros(training_set.output().shape)
            self.Forwarding(training_set.input(), training_set.input().shape[0], output_NN, True)
            loss_training = LOSS(output_NN, training_set.output(), training_set.output().shape[0])
            loss_array_training=np.append(loss_array_training,loss_training)

            #loss on all validation set
            output_val, loss_validation = self.validation(validation_set)
            loss_array_validation=np.append(loss_array_validation, loss_validation)

            if self.type_problem == "classification":
                #accuracy training
                acc_tr=accuracy(self.fun_out,training_set.output(),output_NN)
                accuracy_array_training=np.append(accuracy_array_training,acc_tr)            
                
                #accuracy validation
                acc_vl = accuracy(self.fun_out, validation_set.output(), output_val)
                accuracy_array_validation=np.append(accuracy_array_validation,acc_vl)  
            

            if loss_validation < best_loss_validation:
                best_loss_validation = loss_validation
                best_model = self.struct_layers
            
            #early stopping 
            if self.early_stopping:
                if fun_early_stopping(loss_validation, best_loss_validation):
                    validation_stop -= 1 
                else:
                    validation_stop = 3
                    
            if(validation_stop==0 | (math.isnan(best_loss_validation)) | (math.isnan(best_loss_validation)) | 
               (math.isnan(loss_training)) | (math.isnan(loss_training))):
                print("Early stopping\n")
                break
        
        self.struct_layers = best_model
        
        #########################
        ######## GRAPHS #########
        #########################
        title = "epoch:"+str(epochs)+"; batch:"+str(batch_size)+"; alfa:"+str(self.alfa)+"; lamda:"+str(self.v_lambda)+"\n eta:"+str(self.learning_rate)+"; layer:"+str(self.units)+ "; function:"+str(self.function)+"; function Output_L: linear\nweight:"+str(self.type_weight)
        file_ls='figure/training'+str(num_training)
        file_acc='figure/accuracy'+str(num_training)
        
        if(self.type_problem=="classification"):
            title += "; acc: "+ str(int(acc_tr)) + "; acc_val: " + str(int(acc_vl))    
            graph (title,"Accuracy", accuracy_array_training,accuracy_array_validation,file_acc)
        graph(title,"Loss", loss_array_training,loss_array_validation,file_ls)

    def backprogation(self, output_NN, training_set_output, batch_size):
        for i in range(np.size(self.struct_layers) - 1, -1, -1):
            layer = self.struct_layers[i]
            delta = np.empty([layer.nj,batch_size],float)
           
            #for all nodes of current layer
            for j in range(0, layer.nj):
                nets_batch = layer.net_matrix(j)
                #outputlayer
                if i == (np.size(self.struct_layers) - 1):
                        delta[j,:] = bp._delta_output_layer(training_set_output[:,j], output_NN[:,j], nets_batch, self.fun_out)
                #hiddenlayer
                else:
                    delta[j, :] = bp._delta_hidden_layer(delta_layer_succesivo.T, self.struct_layers[i + 1].w_matrix[j, :], nets_batch, self.function)
                
                gradient = -np.dot(delta[j,:],layer.x)
                gradient = np.divide(gradient,batch_size)
                
                #thikonov regularization
                regularizer=np.dot(self.v_lambda,layer.w_matrix[:, j])
                regularizer[regularizer.shape[0]-1]=0
                
                #momentum
                momentum = self.alfa*layer.Delta_w_old[:,j]

                #update weights
                layer.w_matrix[:, j],  layer.Delta_w_old[:,j] = bp.update_weights(layer.w_matrix[:, j], self.learning_rate, gradient, regularizer, momentum)
            delta_layer_succesivo = delta

    ####################
    # VALIDATION PHASE #
    ####################
    def validation(self,validation_set):
        validation_set_input = validation_set.input()
        validation_set_output = validation_set.output()
        
        output_NN = np.zeros(validation_set_output.shape)
        
        self.Forwarding(validation_set_input, validation_set_input.shape[0], output_NN, True)
        loss_validation = LOSS(output_NN, validation_set_output, validation_set_output.shape[0])

        #mee = MEE(output_NN, validation_set_output, validation_set_output.shape[0])
        
        return output_NN, loss_validation

    def penalty_NN(self):
        penalty = 0
        for layer in self.struct_layers:
            penalty += layer.penalty()
        return self.v_lambda*penalty