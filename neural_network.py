from os import fsdecode
import numpy as np
import Layer
import os
import matplotlib
from graphycs import makegraph
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from function import der_loss, LOSS, _classification,  _derivate_activation_function, sign, accuracy, MEE
from concurrent.futures import ThreadPoolExecutor
import backpropagation as bp

class neural_network:
    
    def __init__(self, nj, alfa, v_lambda, learning_rate, numero_layer, function, fun_out, type_weight, type_problem = "Regression"):

        self.alfa = alfa
        self.v_lambda = v_lambda
        #controlliamo se bisogna usare learning rate variabile
        if (learning_rate >= 1) | (learning_rate == 0):
            self.tau = np.power(2, learning_rate)
        else:
            self.learning_rate=learning_rate
            self.tau = 0
        self.nj=nj
        self.function=function
        self.fun_out = fun_out
        #creo la struttura struct_layer che conterrà i vari layer
        self.struct_layers = np.empty(numero_layer, Layer.Layer)
        self.numero_layer = numero_layer
        self.type_weight=type_weight
        self.type_problem = type_problem
        #inserisco i layer in struct layer
        self.units = []
        for i in range(1,self.numero_layer+1):
            self.struct_layers[i-1]=Layer.Layer(self.nj[i],self.nj[i-1],type_weight)
            self.units.append(nj[i])

    #   --------METODI CHE ERANO NEI THREAD------------
    #restituisce il risultato della moltiplicazione di w*x di tutta la rete e di tutti i layer
    def forward (self, x_input, row_input_layer, validation = False):
        #parto da infondo
        i = np.size(self.struct_layers)-1
        for layer in self.struct_layers:
            x_input = np.append(x_input, 1)
            if validation == False:
                layer.x[row_input_layer,:] = x_input  
            
            #hidden layer
            if i != 0:
                #calcolo il net e lo salvo in x_input
                x_input = layer.output(self.function, x_input)
            
            #output layer
            else:
                output = np.zeros(layer.nj)
                output = layer.output(self.fun_out, x_input)
                
            i = i - 1
        return output

    def task_forwarding(self, x_input, i, output, row_input_layer, validation = False):
        output[i, :] = self.forward(x_input, row_input_layer, validation)
        
    def ThreadPool_Forward(self, matrix_input, index_matrix, batch_size, output, validation = False):
        #creo il pool di thread
        executor = ThreadPoolExecutor(10)
        #indice massimo che posso raggiungere con la dimensione del batch
        max_i = batch_size + index_matrix
       
        #prendo l'indice relativo all'training example del batch
        for i in range(index_matrix, max_i):
            #row_input va da 0 a batch_size-1
            row_input_layer = i % batch_size   
            executor.submit(self.task_forwarding, matrix_input[i, :], i, output, row_input_layer, validation)

        executor.shutdown(True)

        return output
    
    #   --------FINE METODI CHE ERANO NEI THREAD------------
    
    def trainig(self, training_set, validation_set, batch_size, epochs,num_training):
        
        #modifico layer.x in base alla grandezza di batch_size
        for layer in self.struct_layers:
            layer.set_x(batch_size)

        #divido il training in input e output
        training_set_input = training_set.input() 
        training_set_output = training_set.output()

        #print("----------------------MEDIA OUTPUT TRAINING SET ----------------------------------\n", np.sum(training_set_output) / training_set_output.shape[0])
        best_loss_validation= + math.inf
        best_mee = +math.inf
        validation_stop=3

        #variabili per il grafico training
        epoch_training=np.empty(0)
        loss_array=np.empty(0)

        #variabili per il grafico validation
        validation_array=np.empty(0)
        epoch_validation=np.empty(0)
        accuracy_array=np.empty(0)

        for i in range(epochs):
            #divide gli esempi in matrici fatte da batch_size parti
            mini_batch = training_set.create_batch(batch_size)
            #loss sull'intero ciclo di batch
            
            #se usiamo learning rate variabile lo aggiorno
            if self.tau != 0:
                self.learning_rate = 1 / (1 + (i / self.tau))

            for batch in mini_batch:
                #creo array output_NN che mi servirà per memorizzare i risultati di output
                output_NN = np.zeros(batch.output().shape)
                self.ThreadPool_Forward(batch.input(),0, batch_size, output_NN)   
                #----------------------------------questo 0  in backprop non serve!!!!!-----------------------------------------------------------
                self.backprogation(output_NN, batch.output(), batch_size)

            #calcolo la loss su tutto l'intero training
            output_NN = np.zeros(training_set.output().shape)
            self.ThreadPool_Forward(training_set.input(), 0, training_set.input().shape[0], output_NN, True)
            penalty_term = self.penalty_NN()
            loss_training = LOSS(output_NN, training_set.output(), training_set.output().shape[0], penalty_term)
            acc=accuracy(self.type_problem,self.fun_out,training_set.output(),output_NN)
            #salvo nell'array i valori del training
            loss_array=np.append(loss_array,loss_training)
            accuracy_array=np.append(accuracy_array,acc)
            epoch_training=np.append(epoch_training,i)
            
            if (i % 5 == 0):
                #calcolo la loss sulla validazione e me la salvo nell'array
                acc_validation, best_loss_validation, best_mee = self.validation(validation_set, penalty_term)
                validation_array=np.append(validation_array,best_loss_validation)
                epoch_validation=np.append(epoch_validation,i)
       
            #se ho raggiunto lo stesso valore o maggiore del precedente per 3 cicli consecutivi nella validation mi fermo
            # oppure se ho un errore troppo grande o troppo piccolo
            if(validation_stop==0 | (math.isnan(best_loss_validation)) | (math.isnan(best_loss_validation)) | 
               (math.isnan(loss_training)) | (math.isnan(loss_training))):
                print("INTERROMPO!")
                break

        #------------GRAFICO TRAINING-------------------------------
        if(self.type_problem=="classification"):
            acc=accuracy( self.type_problem,self.fun_out,training_set.output(),output_NN)
            titolo= "epoch:"+str(epochs)+"; batch:"+str(batch_size)+"; alfa:"+str(self.alfa)+"; lamda:"+str(self.v_lambda)+"\n eta:"+str(self.learning_rate)+"; layer:"+str(self.units)+ "; function:"+str(self.function)+"; function Output_L:"+str(self.fun_out)+ "\nweight:"+str(self.type_weight) + "; acc: "+ str(int(acc*100)) + "; acc_val: " + str(int(acc_validation*100))
        else:
            titolo= "epoch:"+str(epochs)+"; batch:"+str(batch_size)+"; alfa:"+str(self.alfa)+"; lamda:"+str(self.v_lambda)+"\n eta:"+str(self.learning_rate)+"; layer:"+str(self.units)+ "; function:"+str(self.function)+"; function Output_L: linear\nweight:"+str(self.type_weight)
        
        file='figure/training'+str(num_training)
        makegraph (titolo,epoch_training,accuracy_array,loss_array,self.type_problem,epoch_validation,validation_array,file)

    def backprogation(self, output_NN, training_set_output, batch_size):
        #parto dall'ultimo livello fino ad arrivare al primo
        for i in range(np.size(self.struct_layers) - 1, -1, -1):
            layer = self.struct_layers[i]
            delta = np.empty([layer.nj,batch_size],float)
            der_sig=np.empty(batch_size,float)
           
            #per ogni nodo di ogni layer
            for j in range(0, layer.nj):
                nets_batch = layer.net_matrix(j)
                #outputlayer
                if i == (np.size(self.struct_layers) - 1):
                        delta[j,:] = bp._delta_output_layer(training_set_output[:,j], output_NN[:,j], nets_batch, self.fun_out)
                #hiddenlayer
                else:
                    delta[j, :] = bp._delta_hidden_layer(delta_layer_succesivo.T, self.struct_layers[i + 1].w_matrix[j, :], nets_batch, self.function)
                    #der_sig = _derivate_activation_function(layer.net_matrix(j),self.function)
                    #product è un vettore delta*pesi
                    #product = delta_out.T.dot(self.struct_layers[i + 1].w_matrix[j, :])
                    #delta=Sommatoria da 0 a batch_size di delta_precedenti*pesi
                    #for k in range(batch_size):
                        #delta[j,k]=np.dot(product[k],der_sig[k])
                
                gradient = -np.dot(delta[j,:],layer.x)
                gradient = np.divide(gradient,batch_size)
                #regolarizzazione di thikonov
                regularizer=np.dot(self.v_lambda,layer.w_matrix[:, j])
                regularizer[regularizer.shape[0]-1]=0
                
                #momentum
                momentum = self.alfa*layer.Delta_w_old[:,j]

                #update weights
                layer.w_matrix[:, j],  layer.Delta_w_old[:,j] = bp.update_weights(layer.w_matrix[:, j], self.learning_rate, gradient, regularizer, momentum)
                #regolarizzazione di thikonov
                #temp=np.dot(self.v_lambda,layer.w_matrix[:, j])*2
                #temp[temp.shape[0]-1]=0
                #gradient = gradient - temp
                
                #Dw_new = np.dot(gradient, self.learning_rate)
                #momentum
                #d_new=d_new+alfa*delta_old
                #Dw_new = np.add(Dw_new, np.dot(self.alfa, layer.Delta_w_old[:,j]))
                #layer.Delta_w_old[:,j] = Dw_new
                #update weights
                #layer.w_matrix[:, j] = np.add(layer.w_matrix[:, j], Dw_new)
            delta_layer_succesivo = delta

   # def DeltaW_new(self,Dw_new,D_w_old):
    #    return np.add(Dw_new, np.dot(self.alfa, D_w_old))
    '''

    def backprogation(self, index_matrix, output_NN, training_set_output, batch_size):
        delta_layer_succesivo = np.empty([0,0])
        #matrice dimensione = nodi layer corrente X numero esempi
        delta_layer_corrente = []
        #parto dall'ultimo livello fino ad arrivare al primo
        for i in range(np.size(self.struct_layers) - 1, -1, -1):
            layer = self.struct_layers[i]
            #per ogni nodo di ogni layer
            delta_layer_corrente = np.empty([layer.nj, batch_size])
            #delta_layer_corrente_media = []
            gradient_tot = 0
            for j in range(0, layer.nj):
                for num_example in range(batch_size):
                    #outputlayer
                    if i == (np.size(self.struct_layers) - 1):
                        delta = bp._delta_output_layer(training_set_output[num_example][j], output_NN[num_example][j], layer.net_matrix(j)[num_example], self.fun_out)
                    #hiddenlayer
                    else:
                        delta = bp._delta_hidden_layer(delta_layer_succesivo[:, num_example ],self.struct_layers[i + 1].w_matrix[j, :], layer.net_matrix(j)[num_example], self.function)
                        
                    delta_layer_corrente[j,num_example] = delta

                    gradient = bp.gradiente(delta, layer.x[num_example, :])
                    gradient_tot += gradient
                #delta_layer_corrente_media.append(np.average(delta_layer_corrente))

                gradient_tot = gradient_tot / batch_size
                #regolarizzazione di thikonov
                regularizer=np.dot(self.v_lambda,layer.w_matrix[:, j])
                regularizer[regularizer.shape[0]-1]=0
                
                #momentum
                momentum = self.alfa*layer.Delta_w_old[:,j]

                #update weights
                layer.w_matrix[:, j],  layer.Delta_w_old[:,j] = bp.update_weights(layer.w_matrix[:, j], self.learning_rate, gradient_tot, regularizer, momentum)

            delta_layer_succesivo = delta_layer_corrente
    '''
    #return TRUE if this is the best model
    def validation(self,validation_set, penalty_term = 0):
        validation_set_input = validation_set.input()
        validation_set_output = validation_set.output()
        
        output_NN = np.zeros(validation_set_output.shape)
        
        self.ThreadPool_Forward(validation_set_input, 0, validation_set_input.shape[0], output_NN, True)
        loss_validation = LOSS(output_NN, validation_set_output, validation_set_output.shape[0], penalty_term)
        acc = accuracy(self.type_problem,self.fun_out,validation_set_output, output_NN)
        mee = MEE(output_NN, validation_set_output, validation_set_output.shape[0])
        
        return acc, loss_validation, mee

    def penalty_NN(self):
        penalty = 0
        for layer in self.struct_layers:
            penalty += layer.penalty()
        return self.v_lambda*penalty