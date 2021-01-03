import matplotlib.pyplot as plt
def grafico (x, y, stringx, stringy):
    title = stringx + "-" + stringy
    plt.title(title)
    plt.xlabel(stringx)
    plt.ylabel(stringy)
    plt.plot(x, y)
    plt.show()

