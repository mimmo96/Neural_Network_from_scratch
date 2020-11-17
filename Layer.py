import normalized_initialization as norm_init
import numpy as np
class Layer:
    def __init__(self, x, nj, nj_plus, dim_matrix):
        self.x = np.array(x)
        self.w_matrix = norm_init.init_w(nj, nj_plus, dim_matrix)
