from concurrent.futures import ThreadPoolExecutor
import concurrent
from function import output_nn
from time import sleep
import numpy as np
import time


def task(struct_layers, x_input, i, output):
    output[i, :] = output_nn(struct_layers, x_input)
   # print("output thread: ", i, output[i,:])


def ThreadPool(struct_layers, matrix_input, index_matrix, batch_size, output):
    executor = ThreadPoolExecutor(10)
    max_i = batch_size + index_matrix
    #task(struct_layers, matrix_input[1, :], 1, output)
    #task(struct_layers, matrix_input[2, :], 2, output)
    #task(struct_layers, matrix_input[3, :], 3, output)
    #task(struct_layers, matrix_input[4, :], 4, output)
    for i in range(index_matrix, max_i):
      executor.submit(task, struct_layers, matrix_input[i, :], i, output)
    executor.shutdown(True)

    return output
