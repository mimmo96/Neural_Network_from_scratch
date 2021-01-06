class Matrix_io:
    def __init__(self, matrix, columns_output):
        self.matrix_input = matrix[:, 0:matrix.shape[1] - columns_output]
        self.matrix_output = matrix[:, matrix.shape[1] - columns_output: matrix.shape[1]]
        self.columns_output = columns_output
    
    def input(self):
        return self.matrix_input
    def output(self):
        return self.matrix_output
    
    
    
    
    
    
    def set_input(self, new_matrix_input):
        if self.matrix_input.shape == new_matrix_input.shape:
            self.matrix_input = new_matrix_input
        else:
            raise ("self.matrix_input.shape != new_matrix_input.shape")
    def set_output(self, new_matrix_output):
        if self.matrix_output.shape == new_matrix_output.shape:
            self.matrix_output = new_matrix_output
        else:
            raise ("self.matrix_output.shape != new_matrix_output.shape")