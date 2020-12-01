from concurrent.futures import ThreadPoolExecutor
import concurrent
from time import sleep
from function import output_nn
import numpy as np  
        
def task(i):
    sleep(2)
    return i
    
def ThreadPool(num_righe):
   executor = ThreadPoolExecutor(10)
   output = np.empty(10, concurrent.futures.Future)
   out = np.empty(num_righe, int)
   for i in range(num_righe-10):
        #output[i] = executor.submit(output_nn, struct_layers, matrix_input[i,:])
        for j in range(10):
            output[j] = executor.submit(task, i)
            i = i+1
          #  future = executor.submit(task, i)
        i = i - 10
        for j in range(10):
            out[i+j] = output[j].result()
            print(out[i+j])
            i = i+1

   return out

print(ThreadPool(100))

