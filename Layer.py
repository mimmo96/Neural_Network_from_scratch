from function import init_w
import numpy as np
import function as sig

class Layer:
    #x numero di input 
    # #nj=numero di nodi
    #nj_plus nodi livello successivo

    def __init__(self, nj, nj_prec, intervallo, batch_size = 0):
        self.nj = nj
        self.x = np.empty([batch_size,nj_prec +1],float)
        self.w_matrix = init_w(intervallo, [nj_prec+1,nj])
        self.Delta_w_old = np.zeros(self.w_matrix.shape)

    def net(self, net_i, x_input):
        #if np.size(x_input) != np.size(self.w_matrix[:, net_i]):
         #   print(net_i, x_input, self.w_matrix[:, net_i])
        return np.dot(self.w_matrix[:, net_i], x_input)

    def net_matrix(self, nodo_i):
        return np.dot(self.x, self.w_matrix[:, nodo_i])

    def output(self,x_input):
        out = np.empty(self.nj)
        #i = 0
        for i in range(self.nj):
            net_i = self.net(i, x_input)
            out[i] = sig.sigmoid(net_i)
        return out
    
    def set_x(self, rows):
        self.x = np.empty([rows, self.w_matrix.shape[0]])
