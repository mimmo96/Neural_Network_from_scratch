from function import init_w
import numpy as np
import function as sig

class Layer:
    #x number of inputs
    #units = number of nodes
    #units_plus nodes next level

    def __init__(self, units, units_prec, type_weight, batch_size = 0):
        self.units = units
        self.x = np.empty([batch_size,units_prec +1],float)
        self.w_matrix = init_w( [units_prec+1,units],type_weight)
        self.Delta_w_old = np.zeros(self.w_matrix.shape)

    #net=W*X_input (single component)
    def net(self, net_i, x_input):
        return np.dot(self.w_matrix[:, net_i], x_input)
    
    #net=W*X_input of the whole array
    def net_matrix(self, nodo_i):
        return np.dot(self.x, self.w_matrix[:, nodo_i])

    def output(self, fun, x_input):
        out = np.empty(self.units)
        for i in range(self.units):
            net_i = self.net(i, x_input)
            out[i] = sig.choose_function(fun, net_i)
        return out
    
    def set_x(self, rows):
        self.x = np.empty([rows, self.w_matrix.shape[0]])

    def penalty(self):
        penalty = np.power(self.w_matrix,2)
        penalty = np.sum(penalty, axis=0)
        penalty = np.sum(penalty)
        return penalty
