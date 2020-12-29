import numpy as np
import Layer
import matplotlib.pyplot as plt
from function import output_nn, der_loss, derivate_sigmoid, derivate_sigmoid_2 
from ThreadPool import ThreadPool

class neural_network:

    def __init__(self,dim_input,layer,nj, alfa, v_lambda, epochs, learning_rate, 
                    matrix_in_out, num_input, batch_size,output_expected):

        self.dim_input=dim_input
        self.layer=layer
        self.alfa = alfa
        self.v_lambda = v_lambda
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.matrix_in_out=matrix_in_out
        self.num_input=num_input
        self.batch_size=batch_size
        self.output_expected=output_expected
        self.nj=nj

        #creo la nuova struttura che conterrà i layer
        self.struct_layers = np.empty(self.layer,Layer.Layer)
        for i in range(1,np.size(self.struct_layers)+1):
            self.struct_layers[i-1]=Layer.Layer(nj[i],nj[i-1],nj[i+1],batch_size)


    def mini_batch(self):

        num_righe, num_colonne = self.matrix_in_out.shape
        last_layer = np.size(self.struct_layers) - 1
        num_output_layer = self.struct_layers[last_layer].nj
        output_NN = np.zeros([num_righe, num_output_layer])

        plt.title("grafico")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        epo=[]
        lo=[]

        for i in range(self.epochs):
            index_matrix = np.random.randint(0, (self.num_input - self.batch_size)+1 )
            ThreadPool(self.struct_layers, self.matrix_in_out[:, 0:(num_colonne - 2)], index_matrix, self.batch_size, output_NN)
            loss = self.backprogation(index_matrix,output_NN)
            epo.append(i)
            lo.append(loss)

        plt.plot(epo, lo)
        ThreadPool(self.struct_layers, self.matrix_in_out[:, 0:(num_colonne - 2)], index_matrix, self.batch_size, output_NN)
        print("output ", output_NN)
        print("accuratezza: \n" ,np.abs((self.output_expected -output_NN) / self.output_expected)*100)
        plt.show()

    def backprogation(self,index_matrix,output_NN):
        
        for i in range(np.size(self.struct_layers) - 1, -1, -1):
            # restituisce l'oggetto layer i-esimo
            layer = self.struct_layers[i]
            
            delta = np.empty([layer.nj,self.batch_size],float)
            der_sig=np.empty(self.batch_size,float)

            for j in range(0, layer.nj):
                #outputlayer
                if i == (np.size(self.struct_layers) - 1):
                    max_row = index_matrix+self.batch_size
                    delta[j,:] = der_loss(output_NN[index_matrix:max_row,j], 
                                        self.output_expected[index_matrix:max_row,j]) 
                    #calcolo della loss
                    loss = np.sum(np.subtract(output_NN[index_matrix:max_row,j], 
                                        self.output_expected[index_matrix:max_row,j])) / self.batch_size
                    loss = np.power(loss,2)
                #hiddenlayer
                else:
                    der_sig = derivate_sigmoid_2(layer.net_matrix(j))
                    #product è un vettore delta*pesi
                    product = delta_out.T.dot(self.struct_layers[i + 1].w_matrix[j, :])
                    for k in range(self.batch_size):
                        delta[j,k]=np.dot(product[k],der_sig[k])
                #regolarizzazione di thikonov
                gradient = np.dot(delta[j,:],layer.x) - self.v_lambda*layer.w_matrix[:, j]*2
                gradient = np.divide(gradient,self.batch_size)
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

