import numpy as np


class Matrix_io:
    '''
        this class describe input-output dataset of a Neaural Network
    '''
    def __init__(self, matrix, len_output):
        self.matrix = matrix
        self.len_output = len_output
    
    def input(self):
        return  self.matrix[:, 0:self.matrix.shape[1] - self.len_output]
    
    def output(self):
        return self.matrix[:, self.matrix.shape[1] - self.len_output: self.matrix.shape[1]]
    
    def get_len_output(self):
        return self.len_output        
    def get_len_input(self):
        return self.matrix.shape[1] - self.len_output
    #set new matrix to current
    def set(self, new_matrix):
        if self.matrix.shape == new_matrix.shape:
            self.matrix = new_matrix
        else:
            raise ("self.matrix_input.shape != new_matrix_input.shape")
    
    def create_batch(self, batch_size):
        #array of arrays containing batch_size blocks
        np.random.shuffle(self.matrix)
        mini_batches = []
        #define number of batch
        no_of_batches = self.matrix.shape[0] // batch_size
        for i in range(no_of_batches):
            mini_batch = Matrix_io(self.matrix[i*batch_size:(i+1)*batch_size], self.len_output)
            mini_batches.append(mini_batch)
        
        if self.matrix.shape[0] % batch_size != 0:
            #matrix with the remaining lines of self.matrix
            mini_batch = Matrix_io(self.matrix[(i+1)*batch_size:], self.len_output)
            
    
            mini_batches.append(mini_batch)
        
        return mini_batches
    
    def create_fold(self, start_index, end_index, last_set = False):
        
        if last_set:
            end_index = self.matrix.shape[0]
        
        training_k = np.delete(self.matrix, slice(start_index,end_index), axis=0)
        validation_k = self.matrix[start_index:end_index,:]

        training_k = Matrix_io(training_k,self.len_output )
        validation_k = Matrix_io(validation_k, self.len_output)
        
        return training_k, validation_k 
                
