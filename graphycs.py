import matplotlib.pyplot as plt
import os

#file=patch del file dove devo salvare il file
def graph (title,label_y, array_tr, array_vl, file):
    #grafico training
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(label_y)
    plt.plot(array_tr)

    #grafico validation
    plt.plot(array_vl, "r--")     
    plt.legend(["TRAINING", "TEST"])
    file=file+".png"

    #sovrascrivo il file se esiste già
    if os.path.exists(file):
        os.remove(file)
    plt.savefig(file,format='png',dpi=200)
    plt.close('all')
