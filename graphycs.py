import matplotlib.pyplot as plt
import os

#file=patch del file dove devo salvare il file
def makegraph (titolo,epoch_training,accuracy_array,loss_array,type_problem,epoch_validation,validation_array,file):
    #grafico training
    plt.title(titolo)
    plt.xlabel("Epoch")
    plt.ylabel("LOSS")
    plt.plot(epoch_training,loss_array)

    #grafico validation
    plt.plot(epoch_validation,validation_array, "r--")     
    plt.legend(["TRAINING", "TEST"])
    file=file+".png"

    #sovrascrivo il file se esiste già
    if os.path.exists(file):
        os.remove(file)
    plt.savefig(file,format='png',dpi=200)
    plt.close('all')

     #--------------GRAFICO ACCURATEZZA-----------------------------
    if(type_problem=="classification"):
        #grafico training
        plt.title(titolo)
        plt.xlabel("Epoch")
        plt.ylabel("ACCURATEZZA")
        plt.plot(epoch_training,accuracy_array)

        file=file+"-accuratezza"+".png"
        plt.savefig(file,format='png',dpi=200)
        plt.close('all')
        
