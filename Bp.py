import numpy as np
from function import output_nn, der_loss, derivate_sigmoid, derivate_sigmoid_2 
from ThreadPool import ThreadPool
alfa = 0.9
#v_lambda = 0.01

def backprogation(struct_layers, learning_rate, index_matrix, batch_size, output_expected,output_NN):
 for i in range(np.size(struct_layers) - 1, -1, -1):
    layer = struct_layers[i]
    delta = np.zeros([layer.nj, batch_size], float)
    for j in range(0, layer.nj):
        Dw_new = 0 
        gradient = 0
        for num_ex in range(batch_size):
            if i == (np.size(struct_layers) - 1):
                # delta=2*(y-o)
                delta[j,num_ex] = np.subtract(output_NN[num_ex+index_matrix,j], 
                            output_expected[num_ex+index_matrix,j])
            #hidden layer
            else:
                der_sig = derivate_sigmoid(layer.net(j,layer.x[num_ex,:]))
                delta[j, num_ex] = np.dot(delta_out[:, num_ex], (struct_layers[i + 1].w_matrix[j,:]))
                delta[j, num_ex] = np.dot(delta[j, num_ex], der_sig)
            gradient = gradient + np.dot(delta[j,num_ex], layer.x[num_ex,:])
        Dw_new = np.dot(-gradient, learning_rate) / batch_size
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

    print(output_NN)