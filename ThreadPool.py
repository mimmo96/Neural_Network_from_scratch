from concurrent.futures import ThreadPoolExecutor
import concurrent
from function import output_nn
from time import sleep
import numpy as np
import time
        
def task(struct_layers, x_input, i, output):
   output[i,:] = output_nn(struct_layers, x_input)
   print("output thread: " , i, output[i,:])

    
def ThreadPool(struct_layers, matrix_input, index_matrix, batch_size, output):
   executor = ThreadPoolExecutor(1)
   for i in range(batch_size):
   
      executor.submit(task, struct_layers, matrix_input[i, :], i, output)
   
   #executor.shutdown(True)
   #print("qui",output,"fine")
   return output
  