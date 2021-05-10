import matplotlib.pyplot as plt
import os

#########
# GRAPH #
#########

#file = patch of the file where I have to save the file
def graph (title,label_y, array_tr, array_vl, file):
    #grafico training
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(label_y)
    plt.plot(array_tr)

    #plot validation graph
    plt.plot(array_vl, "r--")     
    plt.legend(["TRAINING", "TEST"])
    file=file+".png"
    plt.ylim(0, 10)
    #overwrite the file if it already exists
    if os.path.exists(file):
        os.remove(file)
    plt.savefig(file,format='png',dpi=200)
    plt.close('all')
