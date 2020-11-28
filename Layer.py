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
        self.x = np.array(x)
        self.w_matrix = init_w(nj, nj_plus, dim_matrix)
        self.Delta_w_old = self.w_matrix


    def net(self, net_i):
        return np.dot(self.w_matrix[:, net_i], self.x)

    def output(self):
        out =np.empty(self.nj)
        for i in range(self.nj):
            net_i = self.net(i)
            out[i] = sig.sigmoid(net_i)
        return out
