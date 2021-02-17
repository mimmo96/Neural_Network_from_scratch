import numpy as np
import Layer
import matplotlib.pyplot as plt
from function import der_loss, derivate_sigmoid_2, MSE
import graphycs
from concurrent.futures import ThreadPoolExecutor
import Matrix_io

class neural_network:
    
    def __init__(self, nj, alfa, v_lambda, learning_rate, numero_layer):

        self.alfa = alfa
        self.v_lambda = v_lambda
        self.learning_rate=learning_rate
        self.nj=nj
        #creo la struttura struct_layer che conterrà i vari layer
        self.struct_layers = np.empty(numero_layer, Layer.Layer)
        self.numero_layer = numero_layer
       
        #inserisco i layer in struct layer
        for i in range(1,self.numero_layer+1):
            self.struct_layers[i-1]=Layer.Layer(self.nj[i],self.nj[i-1],0.7)

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
        best_min_err_validation = -1
        errors_validation = []
        index_matrix = 1
        best_w = 0
        loss_epochs=0;

        #variabili per il grafico training
        epoch_array=np.empty(0)
        loss_array=np.empty(0)

        #variabili per il grafico validation
        validation_error=np.empty(0)
        epoch_validation=np.empty(0)

        for i in range(epochs):
            #divide gli esempi in matrici fatte da batch_size parti
            mini_batch = training_set.create_batch(batch_size)
            dim=np.size(mini_batch)*2 
            #loss sull'intero ciclo di batch
            #print("---------------")

            for batch in mini_batch:
                #creo array output_NN che mi servirà per memorizzare i risultati di output
                output_NN = np.zeros(batch.output().shape)
                self.ThreadPool_Forward(batch.input(),0, batch_size, output_NN) 
                #somma loss di tutto il batch 
                self.backprogation(0, output_NN, batch.output(), batch_size)
            
            '''
            loss_batch=np.divide(loss_batch,dim)
            loss_array=np.append(loss_array,loss_batch)
            epoch_array=np.append(epoch_array,i)
            # print("epoch: ",i," loss:",loss_batch)
            loss_epochs=loss_batch
            '''
            if (i % 5 == 0):
                best_min_err_validation = self.validation(best_min_err_validation, best_w, errors_validation, validation_set)
                validation_error=np.append(validation_error,best_min_err_validation)
                epoch_validation=np.append(epoch_validation,i)
       
        '''
        #grafico training
        plt.title("LOSS/EPOCH")
        plt.xlabel("Epoch")
        plt.ylabel("LOSS")
        plt.plot(epoch_array,loss_array)

        #grafico validation
        plt.plot(epoch_validation,validation_error)     
        plt.legend(["LOSS TRAINING", "VALIDATION ERROR"])
        plt.show()
        '''
        output_NN = np.zeros(training_set_output.shape)
        self.ThreadPool_Forward(training_set_input, 0, training_set_input.shape[0], output_NN, True)
        print("--------------------------TRAINING ",num_training," RESULT----------------------") 
        print("alfa:",self.alfa, "  lamda:", self.v_lambda, "  learning_rate:",self.learning_rate ,"  nj:",self.nj)
        errore=np.sum(np.abs(training_set_output - output_NN))/training_set_output.shape[0]
        print("errore medio training: \n" ,errore )
        #print("loss: \n",loss_epochs)
        #print("output previsto: \n",training_set_output, "\noutput effettivo: \n", output_NN )
        print("------------------------------FINE TRAINING ",num_training,"-----------------------------")
        
        '''
        output_NN = np.zeros(validation_set.output().shape)
        self.ThreadPool_Forward(validation_set.input(), validation_set.input().shape[0], output_NN, True)
        print("----------------------------------------------------------------")
        print( "validation_set", validation_set.output(),"\nerror_best_model", best_min_err_validation, "\nbest model ", output_NN)
        print("----------------------------------------------------------------")
        '''
        return errore,loss_epochs

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
                    max_row = index_matrix+batch_size
                    #delta=(d-o)
                    delta[j,:] = der_loss( output_NN[index_matrix:max_row,j],training_set_output[index_matrix:max_row,j] )
                    #calcolo della loss
                    #sommatoria loss di ogni neurone (d-o)^2
                    #loss = loss + MSE(output_NN[index_matrix:max_row,j],training_set_output[index_matrix:max_row,j], batch_size)
                #hiddenlayer
                else:
                    #calcola il net di tutta la matrice e applica la funzione
                    #sig(net)
                    der_sig = derivate_sigmoid_2(layer.net_matrix(j))
                    #product è un vettore delta*pesi
                    product = delta_out.T.dot(self.struct_layers[i + 1].w_matrix[j, :])
                    #delta=Sommatoria da 0 a batch_size di delta_precedenti*pesi
                    for k in range(batch_size):
                        delta[j,k]=np.dot(product[k],der_sig[k])
                #regolarizzazione di thikonov
                gradient = -np.dot(delta[j,:],layer.x) - np.dot(self.v_lambda,layer.w_matrix[:, j])*2
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