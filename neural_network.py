import numpy as np
import Layer
import matplotlib.pyplot as plt
import math
from function import der_loss, LOSS, _classification,  _derivate_activation_function
import graphycs
from concurrent.futures import ThreadPoolExecutor
import Matrix_io

class neural_network:
    
    def __init__(self, nj, alfa, v_lambda, learning_rate, numero_layer,function,type_weight, type_problem = "Regression"):

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
        #creo la struttura struct_layer che conterrà i vari layer
        self.struct_layers = np.empty(numero_layer, Layer.Layer)
        self.numero_layer = numero_layer
        self.type_weight=type_weight
        self.type_problem = type_problem
        #inserisco i layer in struct layer
        for i in range(1,self.numero_layer+1):
            self.struct_layers[i-1]=Layer.Layer(self.nj[i],self.nj[i-1],type_weight,function)

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
        errors_validation = []
        best_loss_validation=-1
        validation_stop=3

        #layer migliori sulla validation
        best_struct_layers = 0

        #variabili per il grafico training
        epoch_array=np.empty(0)
        loss_array=np.empty(0)

        #variabili per il grafico validation
        validation_array=np.empty(0)
        epoch_validation=np.empty(0)

        for i in range(epochs):
            #divide gli esempi in matrici fatte da batch_size parti
            mini_batch = training_set.create_batch(batch_size)
            dim=np.size(mini_batch)*2 
            #loss sull'intero ciclo di batch
            #print("---------------")
            
            #se usiamo learning rate variabile lo aggiorno
            if self.tau != 0:
                self.learning_rate = 1 / (1 + (i / self.tau))

            for batch in mini_batch:
                #creo array output_NN che mi servirà per memorizzare i risultati di output
                output_NN = np.zeros(batch.output().shape)
                self.ThreadPool_Forward(batch.input(),0, batch_size, output_NN) 
                #----------------------------------questo 0  in backprop non serve!!!!!-----------------------------------------------------------
                self.backprogation(0, output_NN, batch.output(), batch_size)

            #calcolo la loss su tutto l'intero training
            output_NN = np.zeros(training_set.output().shape)
            self.ThreadPool_Forward(training_set.input(), 0, training_set.input().shape[0], output_NN, True)
            loss_training = LOSS(output_NN, training_set.output(), training_set.output().shape[0],training_set.output().shape[1])
            
            #salvo nell'array i valori del training
            loss_array=np.append(loss_array,loss_training)
            epoch_array=np.append(epoch_array,i)
            
            if (i % 5 == 0):
                #calcolo la loss sulla validazione e me la salvo nell'array
                best_loss_validation = self.validation(best_loss_validation, best_struct_layers, errors_validation, validation_set)
                validation_array=np.append(validation_array,best_loss_validation)
                epoch_validation=np.append(epoch_validation,i)
       
                   #se ho raggiunto lo stesso valore o maggiore del precedente per 3 cicli consecutivi nella validation mi fermo
            # oppure se ho un errore troppo grande o troppo piccolo
            if(validation_stop==0 | (math.isnan(best_loss_validation)) | (math.isnan(best_loss_validation)) | 
               (math.isnan(loss_training)) | (math.isnan(loss_training))):
                break

        '''
        #grafico training
        plt.title("LOSS/EPOCH")
        plt.xlabel("Epoch")
        plt.ylabel("LOSS")
        plt.plot(epoch_array,loss_array)

        #grafico validation
        plt.plot(epoch_validation,validation_array)     
        plt.legend(["LOSS TRAINING", "VALIDATION ERROR"])
        plt.show()
        
        '''
        output_NN = np.zeros(training_set_output.shape)
        self.ThreadPool_Forward(training_set_input, 0, training_set_input.shape[0], output_NN, True)
        print("--------------------------TRAINING ",num_training," RESULT----------------------") 
        print("epoch:",epochs," batch_size:",batch_size," alfa:",self.alfa, "  lamda:", self.v_lambda, "  learning_rate:",self.learning_rate ,"  layer:",self.nj, " function:",self.function, " weight:", self.type_weight)
        #print("errore training: \n" ,errore )
        #print("loss: \n",loss_epochs)
        #print("output previsto: \n",training_set_output, "\noutput effettivo: \n", output_NN )
        
        '''
        output_NN = np.zeros(validation_set.output().shape)
        self.ThreadPool_Forward(validation_set.input(), validation_set.input().shape[0], output_NN, True)
        print("----------------------------------------------------------------")
        print( "validation_set", validation_set.output(),"\nerror_best_model", best_min_err_validation, "\nbest model ", output_NN)
        print("----------------------------------------------------------------")
        '''

    def backprogation(self, index_matrix, output_NN, training_set_output, batch_size):
        #parto dall'ultimo livello fino ad arrivare al primo
        for i in range(np.size(self.struct_layers) - 1, -1, -1):
            layer = self.struct_layers[i]
            delta = np.empty([layer.nj,batch_size],float)
            der_sig=np.empty(batch_size,float)
           
            #per ogni nodo di ogni layer
            for j in range(0, layer.nj):
                #outputlayer
                if i == (np.size(self.struct_layers) - 1):
                    if self.type_problem == "Regression":
                         delta[j,:] = der_loss( output_NN[index_matrix:max_row,j],training_set_output[index_matrix:max_row,j] )
                    else: 
                        delta[j,:] = _classification(layer.net_matrix(j), output_NN[:,j], training_set_output[:,j], self.function)

                    ##################### da commentare ##########################
                    max_row = index_matrix+batch_size
                    #delta=(d-o)
                    delta[j,:] = der_loss( output_NN[index_matrix:max_row,j],training_set_output[index_matrix:max_row,j] )
                    ##############################################################
                #hiddenlayer
                else:
                    der_sig = _derivate_activation_function(layer.net_matrix(j),self.function)
                    #product è un vettore delta*pesi
                    product = delta_out.T.dot(self.struct_layers[i + 1].w_matrix[j, :])
                    #delta=Sommatoria da 0 a batch_size di delta_precedenti*pesi
                    for k in range(batch_size):
                        delta[j,k]=np.dot(product[k],der_sig[k])

                #regolarizzazione di thikonov
           
                temp=np.dot(self.v_lambda,layer.w_matrix[:, j])*2
                temp[temp.shape[0]-1]=0
                gradient = -np.dot(delta[j,:],layer.x) - temp
                gradient = np.divide(gradient,batch_size)
                Dw_new = np.dot(gradient, self.learning_rate)
                #momentum
                #d_new=d_new+alfa*delta_old
                Dw_new = np.add(Dw_new, np.dot(self.alfa, layer.Delta_w_old[:,j]))
                layer.Delta_w_old[:,j] = Dw_new
                #update weights
                layer.w_matrix[:, j] = np.add(layer.w_matrix[:, j], Dw_new)
            delta_out = delta

    def DeltaW_new(self,Dw_new,D_w_old):
        return np.add(Dw_new, np.dot(self.alfa, D_w_old))

    #return TRUE if this is the best model
    def validation(self,best_loss_validation, best_struct_layers, errors_validation, validation_set):
        validation_set_input = validation_set.input()
        validation_set_output = validation_set.output()
        output_NN = np.zeros(validation_set_output.shape)
        self.ThreadPool_Forward(validation_set_input, 0, validation_set_input.shape[0], output_NN, True)
        loss_validation = LOSS(output_NN, validation_set_output, validation_set_output.shape[0],validation_set_output.shape[1])
        errors_validation.append(loss_validation)
        if (loss_validation < best_loss_validation) | (best_loss_validation == -1):
            best_struct_layers = self.struct_layers
            best_loss_validation = loss_validation
        return best_loss_validation