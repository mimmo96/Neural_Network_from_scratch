from concurrent.futures import ThreadPoolExecutor
import concurrent
from function import forward
from time import sleep
import numpy as np
import time


def task(struct_layers, x_input, i, output, row_input_layer, validation = False):
    output[i, :] = forward(struct_layers, x_input, row_input_layer, validation)

def ThreadPool(struct_layers, matrix_input, index_matrix, batch_size, output, validation = False):
    executor = ThreadPoolExecutor(10)
    max_i = batch_size + index_matrix
    for i in range(index_matrix, max_i):
        row_input_layer = i % batch_size
        executor.submit(task, struct_layers, matrix_input[i, :], i, output, row_input_layer, validation)
    executor.shutdown(True)

    return output
