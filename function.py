import numpy as np
from scipy.special import expit
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import Matrix_io
import pandas
import File_names as fn

####################################
# WEIGHTS INITIALIZATION FUNCTIONS #
####################################
def choose_weight_initialization(type_weight,fan_in,fan_out):
    #best for sigmoid activation function
    if type_weight=="random":
        #[-intervallo,intervallo]
        return np.random.uniform(-0.25, 0.25)

    if type_weight=="uniform":
        #[-1/sqrt(fan-in),1/sqrt(fan-out)]
        a=np.divide(1,np.sqrt(fan_in))
        b=np.divide(1,np.sqrt(fan_out))
        return np.random.uniform(-a, b)

    if type_weight=="Xavier Normal":
        #Wij ~ N(mean,std) , mean=0 , std=sqrt(2/(fan_in + fan_out)).
        std=np.sqrt(2/(fan_in + fan_out))
        dist=np.random.normal(0,std,None)
        return dist
    
    if type_weight=="Xavier Uniform":
        # [-sqrt(6)/sqrt(fan_in+fan_out),sqrt(6)/sqrt(fan_in + fan_out)]
        inter= np.divide(np.sqrt(6),np.sqrt(fan_in+fan_out))
        return np.random.uniform(-inter, inter)
    
    #best for RelU
    if type_weight=="He uniform":
        #Uniform [-sqrt(6/fan_in),sqrt(6/fan_in)]
        inter= np.divide(np.sqrt(6),fan_in)
        return np.random.uniform(-inter, inter)
    
    if type_weight=="He Normal":
        #Wij ~ N(mean,std) , mean=0 , std=sqrt(2/fan_in).
        std=np.sqrt(np.divide(2,fan_in))
        dist=np.random.normal(0,std,None)
        return dist
    
    if type_weight=="Glorot":
        stddev = np.sqrt(np.divide(2, (fan_in+fan_out)))
        dist = np.random.normal(0,stddev,None)
        return dist

    return np.random.uniform(-0.7, 0.7)
    
def init_w( dim_matrix,type_weight):
    w = np.zeros([dim_matrix[0], dim_matrix[1]])
    for i in range(dim_matrix[0]):
        for j in range(dim_matrix[1]):
            while abs(w[i,j]) < 0.0000001:
                w[i, j] = choose_weight_initialization(type_weight,dim_matrix[0],dim_matrix[1])
    return w


############################
#   ACTIVATION FUNCTIONS   #
############################

def _relu (x):
    '''
        REctified Linear Unit acivation function: relu(x) = max(0,x)
    '''
    return np.maximum(0,x)

def _identity (x):
    '''
        identity function: identity(x) = x
    '''
    return x

def _logistic (x):
    '''
        logistic activation function: logistic(x) = 1 / (1 + exp(-x))
    '''
    # return 1 / ( 1 + math.exp(-x) )
    #div=np.divide(1.0,1+np.exp(-x))
    return expit(x)

def _tanh (x):
    '''
        tanh activation function (returns hyperbolic tangent of the input) = tanh(x)
    '''
    return np.tanh (x)

def _zero_one_tanh (x):
    '''
        tanh activation function which output is from zero to one: _zero_one_tanh(x) = (1 + tanh(x))/2
    '''
    return (1 + np.tanh (x))/2


########################################
#   ACTIVATION FUNCTIONS DERIVATIVES   #
########################################

def _relu_derivative (x):
    '''
        REctified Linear Unit activation function derivative: relu'(x) = 0 if x<0
                                                              relu'(x) = 1 if x>=0
    '''
    return (x>=0).view('i1')

def _identity_derivative (x):
    '''
        identity function derivative: identity'(x) = 1
    '''
    return 1

def _logistic_derivative (x):
    '''
        logistic activation function derivative: logistic'(x) = logistic(x) * ( 1 - logistic(x) )
    '''
    return  _logistic(x) * ( 1 - _logistic(x) )

def _tanh_derivative (x):
    '''
        tanh activation function derivatives: tanh'(x) = 1 - (tanh(x))**2
    '''
    return 1 - (np.tanh (x))**2

def _zero_one_tanh_derivative (x):
    '''
        zero-one tanh activation function derivatives: tanh'(x) = 1/2 * ( 1 - (tanh(x))**2 )
    '''
    return 1/2 * ( 1 - (np.tanh (x))**2 )


#####################
# CHOOSING FUNCTION #
#####################

def choose_function(fun,net):
    if fun=="sigmoidal":
        return _logistic(net)
    
    if fun=="tanh":
        return _tanh(net)
    
    if fun=="relu":
        return _relu(net)
    
    if fun=="identity":
        return _identity(net)
    
    if fun=="zero_one_h":
        return _zero_one_tanh(net)
    if fun == "Regression":
        return net

def choose_derivate_function(fun,net):
    if fun=="sigmoidal":
        return _logistic_derivative(net)
    
    if fun=="tanh":
        return _tanh_derivative(net)
    
    if fun=="relu":
        return _relu_derivative(net)
    
    if fun=="identity":
        return _identity_derivative(net)
    
    if fun=="zero_one_h":
        return _zero_one_tanh_derivative(net)
   
    if fun == "Regression":
        return 1

def _derivate_activation_function(nets,type_fun):
    sig=np.empty(np.size(nets))
    i=0
    for net in nets:
        sig[i] = choose_derivate_function(type_fun,net)
        i=i+1
    return sig


##################
# LOSS FUNCTIONS #
##################

def der_loss( output_expected,output_layer):
    val = np.subtract(output_expected,output_layer)
    return val

def LOSS(output, output_expected, penalty_term=0):
    mse=np.mean( np.square( output-output_expected ) )/2 
    mse += penalty_term
    return mse

def MEE(output, output_expected):
    #norma euclidea
    batch_size = output.shape[0]
    squares = (np.subtract(output,output_expected)) ** 2    
    squares=np.sum (squares, axis=1)
    squares=np.sqrt (squares)
    squares=np.sum (squares, axis=0)
    return squares/batch_size


#####################
# ACCURACY FUNCTION #
#####################

def accuracy(fun,output_expected, output_NN):

    output_NN=sign(fun,output_NN)
    diff = output_expected - output_NN 
    #diff = np.sum(diff,axis=1)
    count = 0
    for result in diff:
        if (result==0):
            count += 1
    
    return (np.divide(count,np.size(output_expected)))*100

def sign(fun,vector):
    
    for i in range (0,np.size(vector)):
        if fun=="sigmoidal":
            if vector[i] >= 0.5:
                vector[i]= 1
            else:
                vector[i]= 0
            
        if fun=="tanh":
            if vector[i] >= 0:
                vector[i]= 1
            else:
                vector[i]= 0

        if fun=="zero_one_h":
            if vector[i] >= 0.5:
                vector[i]= 1
            else:
                vector[i]= 0

    return vector



###########################
# EARLY STOPPING FUNCTION #
###########################
def fun_early_stopping(err, best_err):
    treshold = 2
    GL = 100* ((err / best_err) - 1)
    if GL >= treshold:
        return True
    else:
        return False


##########################
# NORMALIZATION FUNCTION #
##########################

def normalize_input(x,dim_output):
    colonne = x.shape[1]
    x_input = x[:, 0:(colonne-dim_output)]
    max = x_input.max()
    min = x_input.min()

    x_input = (x_input - min)/(max -min)
    x[:, 0:(colonne-dim_output)] = x_input

    return x


####################
# ONE HOT ENCODING #
####################

def one_hot_encoding(data_set):

    one = OneHotEncoder(sparse=False)
    label=LabelEncoder()
    data_set_X = one.fit_transform(data_set[:,:-1])
    tmp = np.append(data_set_X,np.zeros([len(data_set_X),1]),1)
    tmp[:, -1] = label.fit_transform(data_set[:, -1])
    data_set = tmp 
    return data_set


#############################
# DIVIDE EXAMPLES FUNCTIONS #
#############################
    
def divide_exaples_hold_out(matrix_input, columns_output):
    #divide 70% TR, 20% VL, 10% TS
    rows = matrix_input.shape[0]
    training_size = rows *70 //100
    validation_size = rows *20 //100
    training = Matrix_io.Matrix_io(matrix_input[0:training_size, :], columns_output)
    validation = Matrix_io.Matrix_io(matrix_input[training_size:training_size+validation_size, :], columns_output)
    test = Matrix_io.Matrix_io(matrix_input[training_size+validation_size:, :], columns_output)
    return [training, validation, test]

def setBlind(matrix_input, columns_output):
    blind = Matrix_io.Matrix_io(matrix_input, columns_output)
    return blind


def setPandas():
    df = pandas.DataFrame(columns = ['Number_Model',
                'Units_per_layer',
                'learning_rate' ,
                'lambda',
                'alfa',
                'Function_hidden',
                'inizialization_weights',
                'Error_MSE_tr',
                'Error_MSE_vl',
                'Error_MEE_tr',
                'Error_MEE_vl',
                'Accuracy_tr',
                'Accuracy_vl',
                'Variance'])

    dp = pandas.DataFrame(columns = ['Number_Model',
                    'Error_MSE_ts',
                    'Error_MEE_ts'])

    df.to_csv(fn.top_general_results)
    df.to_csv(fn.general_results)
    dp.to_csv(fn.top_result_test)