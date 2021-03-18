import numpy as np
from function import LOSS

class ensemble:
    
    #NN_array=array contenente le migliroi 10 Neural network
    #data= dati su cui testare
    def __init__(self,NN,data):
        self.NN_array=NN
        self.data=data
    
    #mi restituisce la media degli output della rete sui dati
    def output_average(self):
        output=0
        count=0
        for NN in self.NN_array:
            output_NN = np.zeros(self.data.output().shape)
            NN.ThreadPool_Forward(self.data.input(), 0, self.data.input().shape[0], output_NN, True)
            output=output+output_NN
            count=count+1
        output=np.divide(output,count)
        return output

    #loss carcolata sugli output della rete
    def loss_average(self):
        #calcolo la media degli output, mi restituisce un unico vettore di output
        output=self.output_average()
        loss_test = LOSS(output, self.data.output(), self.data.output().shape[0], penalty_term = 0)
        return loss_test
