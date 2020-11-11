import numpy as np

def init_w(nj, nj_plus, dim_matrix):
    intervallo = np.divide(np.sqrt(6), np.sqrt((nj+nj_plus)))
    w = np.zeros([dim_matrix[0], dim_matrix[1]])
    for i in range(dim_matrix[0]):
        for j in range(dim_matrix[1]):
            while abs(w[i,j]) < 0.0000001:
                w[i, j] = np.random.uniform(-intervallo, intervallo)
        
    return w

