import numpy as np
import matplotlib.pyplot as plt
import math

def init_w(intervallo, dim_matrix):
    #intervallo = np.divide(np.sqrt(6), np.sqrt((nj+nj_plus)))
    w = np.zeros([dim_matrix[0], dim_matrix[1]])
    for i in range(dim_matrix[0]):
        for j in range(dim_matrix[1]):
            while abs(w[i,j]) < 0.0000001:
                w[i, j] = np.random.uniform(-intervallo, intervallo)
    return w

############################
#   ACTIVATION FUNCTIONS   #
############################

def _relu (x):
    '''
        REctified Linear Unit acivation function: relu(x) = max(0,x)
    '''
    return max(0,x)

def _identity (x):
    '''
        identity function: identity(x) = x
    '''
    return x

def _threshold (x):
    '''
        threshold activation function: treshold(x) = 1 if x>0
                                       treshold(x) = 0 if x<=0
    '''
    return 1 if x>0 else 0

def _logistic (x):
    '''
        logistic activation function: logistic(x) = 1 / (1 + exp(-x))
    '''
    return 1 / ( 1 + math.exp(-x) )

def _tanh (x):
    '''
        tanh activation function (returns hyperbolic tangent of the input) = tanh(x)
    '''
    return math.tanh (x)

def _zero_one_tanh (x):
    '''
        tanh activation function which output is from zero to one: _zero_one_tanh(x) = (1 + tanh(x))/2
    '''
    return (1 + math.tanh (x))/2

########################################
#   ACTIVATION FUNCTIONS DERIVATIVES   #
########################################

def _relu_derivative (x):
    '''
        REctified Linear Unit activation function derivative: relu'(x) = 0 if x<0
                                                              relu'(x) = 1 if x>=0
    '''
    return 0 if x<=0 else 1

def _identity_derivative (x):
    '''
        identity function derivative: identity'(x) = 1
    '''
    return 1

def _threshold_derivative (x):
    '''
        threshold activation function derivative: treshold'(x) = 0
    '''
    return 0

def _logistic_derivative (x):
    '''
        logistic activation function derivative: logistic'(x) = logistic(x) * ( 1 - logistic(x) )
    '''
    return _logistic(x) * ( 1 - _logistic (x) )

def _tanh_derivative (x):
    '''
        tanh activation function derivatives: tanh'(x) = 1 - (tanh(x))**2
    '''
    return 1 - (math.tanh (x))**2

def _zero_one_tanh_derivative (x):
    '''
        zero-one tanh activation function derivatives: tanh'(x) = 1/2 * ( 1 - (tanh(x))**2 )
    '''
    return 1/2 * ( 1 - (math.tanh (x))**2 )

# funzione sigmoidale
def sigmoid(x):
    if -x > np.log(np.finfo(type(x)).max):
        return 0.0
    return np.divide(1, np.add(1, np.exp(-x)))

# derivata funzione sigmoidale
def derivate_sigmoid(x):
    sig = sigmoid(x)
    return np.dot(sig, np.subtract(1, sig))

def derivate_sigmoid_2(x):
    sig=np.empty(np.size(x))
    i=0
    for net in x:
        sig[i] = sigmoid(net)
        i=i+1
    return sig

# derivata loss
def der_loss( output_expected,output_layer):
    val = np.subtract(output_expected,output_layer)
  
    return val

def normalize_input(x,dim_output):
    colonne = x.shape[1]
    x_input = x[:, 0:(colonne-dim_output)]
    max = x_input.max()
    min = x_input.min()

    x_input = (x_input - min)/(max -min)
    x[:, 0:(colonne-dim_output)] = x_input
    #print(x)
    return x

def LOSS(output, output_expected, example_cardinality,num_output):
    #(d-o)^2
    mse=np.power(np.subtract(output, output_expected),2)
    mse = np.sum(mse,axis=0)
    mse = np.sum(mse)
    mse = np.divide(mse,np.dot(example_cardinality,num_output)*2)
    return mse

def accuracy(real_matrix, matrix):
    a = real_matrix - matrix
    return np.sum(a, 0)

r = np.matrix([[1, 4, 5], 
    [-5, 8, 9]])
m = np.matrix([[2, 4, 5], 
    [-2, 8, 9]])

#print(accuracy(r, m))

#restituisce un array di matrici
#data: matrice
#batch_size: dimensione di ogni batch
def create_batch(data, batch_size):
    #array di matrici contenente blocchi di dimensione batch_size
    mini_batches = []
    #definisce numero di batch
    no_of_batches = data.shape[0] // batch_size
    for i in range(no_of_batches):
        mini_batch = data[i*batch_size:(i+1)*batch_size]
        mini_batches.append(mini_batch)
    if data.shape[0] / batch_size != 0:
        #matrice con le restanti righe di data
        mini_batch = data[(i+1)*batch_size:]
        if mini_batch.shape[0] < batch_size:
            mini_batch = np.append(mini_batch, data[0:batch_size-mini_batch.shape[0]], axis = 0)
        mini_batches.append(mini_batch)
    return mini_batches

