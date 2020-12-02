from function import init_w
import numpy as np
import function as sig

class Layer:
    #x numero di input 
    # #nj=numero di nodi
    #nj_plus nodi livello successivo
    #dim_matrix dimensione matrice pesi
    def __init__(self, x, nj, nj_plus, dim_matrix):
        self.nj = nj
      #  self.x = np.array(x)
        self.w_matrix = init_w(nj, nj_plus, dim_matrix)
        self.Delta_w_old = self.w_matrix


    def net(self, net_i, x_input):
        print("prodotto net ", np.dot(self.w_matrix[:, net_i], x_input))
        return np.dot(self.w_matrix[:, net_i], x_input)
    def output(self,x_input):
        out = np.empty(self.nj)
        i = 0
        for i in range(self.nj):
            print(i)
            net_i = self.net(i, x_input)
            print("siamo in output", net_i)
            out[i] = sig.sigmoid(net_i)
        return out
