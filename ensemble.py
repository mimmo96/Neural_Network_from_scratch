import numpy as np
from function import LOSS

class ensemble:
    
    def __init__(self,NN,data):
        self.NN=NN
        self.data=data
    
    #mi restituisce un vettore contenente la media degli output
    def output_average(self,output,average):
        output=np.divide(output,average)
        return output

    #vettore output contenente gli output dei diversi modelli della NN
    def loss_average(self,output_NN,average):
        #calcolo la media degli output, mi restituisce un unico vettore di output
        output=self.output_average(output_NN,average)
        penalty_term = self.NN.penalty_NN()
        loss = LOSS(output, self.data, self.data.shape[0],penalty_term)
        return loss
