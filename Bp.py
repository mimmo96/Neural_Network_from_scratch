import numpy as np
from function import output_nn, der_loss, derivate_sigmoid, derivate_sigmoid_2 
from ThreadPool import ThreadPool
alfa = 0.9
v_lambda = 0.01

def backprogation(struct_layers, learning_rate, index_matrix, batch_size, output_expected,output_NN):
    for i in range(np.size(struct_layers) - 1, -1, -1):
        # restituisce l'oggetto layer i-esimo
        layer = struct_layers[i]
        
        delta = np.empty([layer.nj,batch_size],float)
        der_sig=np.empty(batch_size,float)

        for j in range(0, layer.nj):
            #outputlayer
            if i == (np.size(struct_layers) - 1):
                max_row = index_matrix+batch_size
                delta[j,:] = der_loss(output_NN[index_matrix:max_row,j], 
                                    output_expected[index_matrix:max_row,j])
            #hiddenlayer
            else:
                der_sig = derivate_sigmoid_2(layer.net_matrix(j))
                #product è un vettore delta*pesi
                product = delta_out.T.dot(struct_layers[i + 1].w_matrix[j, :])
                for k in range(batch_size):
                    delta[j,k]=np.dot(product[k],der_sig[k])
            delta[j,:] = delta[j,:] 
            gradient = np.dot(delta[j,:],layer.x) + v_lambda*layer.w_matrix[:, j]*2
            gradient = np.divide(gradient,batch_size)
            Dw_new = np.dot(gradient, learning_rate)
            #print("gradiente i:",i," j:",j," -> ",Dw_new)
            layer.w_matrix[:, j] = np.add(layer.w_matrix[:, j], Dw_new)
        delta_out = delta

def DeltaW_new(Dw_new,D_w_old, it):
    if it != 0:
       return np.subtract(Dw_new, np.dot(alfa, D_w_old))
    return Dw_new


def minbetch(struct_layers, epochs, learning_rate, matrix_in_out, num_input, batch_size,output_expected):
    num_righe, num_colonne = matrix_in_out.shape
    last_layer = np.size(struct_layers) - 1
    num_output_layer = struct_layers[last_layer].nj
    output_NN = np.zeros([num_righe, num_output_layer])

    for i in range(epochs):
        index_matrix = 0#np.random.randint(0, (num_input - batch_size)+1 )
        ThreadPool(struct_layers, matrix_in_out[:, 0:(num_colonne - 2)], index_matrix, batch_size, output_NN)
        backprogation(struct_layers, learning_rate,index_matrix,batch_size,output_expected,output_NN)

    ThreadPool(struct_layers, matrix_in_out[:, 0:(num_colonne - 2)], index_matrix, batch_size, output_NN)
    print(output_NN)