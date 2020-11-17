import normalized_initialization as norm_init
import numpy as np
import sigmoidal as sig


class Layer:
    def __init__(self, x, nj, nj_plus, dim_matrix):
        self.nj = nj
        self.x = np.array(x)
        self.w_matrix = norm_init.init_w(nj, nj_plus, dim_matrix)

    def net(self, net_i):
        return np.dot(self.w_matrix[:, net_i], self.x)

    def output(self):
        out =np.empty(self.nj)
        for i in range(self.nj):
            net_i = self.net(i)
            print(net_i)
            out[i] = sig.sigmoid(net_i)
        return out
