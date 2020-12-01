import numpy as np
from function import output_nn, der_loss, derivate_sigmoid
alfa = 0.9
#v_lambda = 0.01

def backprogation(struct_layers, num_epoch, learning_rate, x_input, output_expected):
    for epoch in range(0, num_epoch):
        output_NN = output_nn(struct_layers, x_input)
        # scandisce tutti i layer presenti all'interno della rete
        for i in range(np.size(struct_layers) - 1, -1, -1):
            # restituisce l'oggetto layer i-esimo
            layer = struct_layers[i]
            # if output layer
            delta = np.zeros(layer.nj)

            for j in range(0, layer.nj):
                #outputlayer
                if i == (np.size(struct_layers) - 1):    
                    delta[j] = der_loss(output_NN[j], output_expected[j])

                #hiddenlayer
                else:
                    der_sig = derivate_sigmoid(layer.net(j))
                    #(delta_livello_successivo) * w del nodo_corrente(j) ai nodi successivi(k)
                    product = np.dot(struct_layers[i + 1].w_matrix[j, :], delta_out)
                    delta[j] = np.sum(product)
                    delta[j] = np.dot(delta[j], der_sig)

                gradient = np.dot(layer.x, delta[j])
                Dw_new = np.dot(gradient, learning_rate)
                #Dw_new = DeltaW_new(Dw_new, layer.Delta_w_old[:,j], epoch)
                #tmp = layer.w_matrix[:, j]
                layer.w_matrix[:, j] = np.add(layer.w_matrix[:, j], Dw_new)
                #layer.w_matrix[:, j] = np.subtract(layer.w_matrix[:, j], np.dot(v_lambda,tmp))
                #layer.Delta_w_old[:,j] = Dw_new

            delta_out = delta

    print(np.subtract(output_expected, output_NN))
    print(output_NN, output_expected)

def DeltaW_new(Dw_new,D_w_old, it):
    if it != 0:
       return np.subtract(Dw_new, np.dot(alfa, D_w_old))
    return Dw_new


def minbetch(struct_layers, epochs, learning_rate, matrix_input, num_input, batch_size):
    for i in range(epochs):
        index_matrix=np.random.uniform(0, num_input-batch_size)
        