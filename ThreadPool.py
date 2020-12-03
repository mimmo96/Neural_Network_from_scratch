from concurrent.futures import ThreadPoolExecutor
import concurrent
from function import output_nn
from time import sleep
import numpy as np
import time


def task(struct_layers, x_input, i, output, row_input_layer):
    output[i, :] = output_nn(struct_layers, x_input, row_input_layer)
   # print("output thread: ", i, output[i,:])


def ThreadPool(struct_layers, matrix_input, index_matrix, batch_size, output):
    executor = ThreadPoolExecutor(10)
    max_i = batch_size + index_matrix
    for i in range(index_matrix, max_i):
        row_input_layer = i % batch_size
        executor.submit(task, struct_layers, matrix_input[i, :], i, output, row_input_layer)
        
    executor.shutdown(True)

    return output
