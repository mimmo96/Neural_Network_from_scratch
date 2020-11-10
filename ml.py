import numpy as np


m= np.array(  [ [1,2,3],[3,4,6],[7,8,9] ] )

#numero di elementi nella matrice (dimensione)
print(m.size)
#numero di righe e numero di colonne (dimensione)
m.shape

m[0,0]
m[0,:]
m[:,0]
print(m[0,:]);
print(m[:,0]);

newM=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

print(newM);

#cambia la dimensione delle matrice da 4x3 a-> 3x4
#newM=newM.reshape(3,4)
newrow = [1,2,3]
newM=np.vstack([newM, [newrow]])
print(newM);

#trasforma matrice in un vettore di dimensione m.size(num el matrice)
m=m.reshape(1,m.size)
print(m)

#trasforma la matrice in un array
m=m.flatten()
print(m)

#matrici con valori 0
print(np.zeros([3,3]))

#matrici con valori 1
print(np.ones([3,3]))

#matrice diagonale
print(np.eye(3,3))

#matrici con valori random (NOTA: sono compresi tra 0 e 1 quindi se io li voglia tra 2 e 3 basta moltiplicarli)
print(np.random.rand(3,3))

#genera valori da 0 a 5 interi
print(np.random.randint(5,size=[3,3]))

#valore medio,deviazione standard,tupla con dimensione matrice da creare 
print(np.random.normal(0,1,(3,3)))

#generiamo un intervallo in un array che va da 0 al nostro valore definito
print(np.arange(10))

#generiamo un intervallo in un array che va da 2 a 9
print(np.arange(2,10))

#operazioni tra matrici
a=np.array([[1,2],[2,4]])
b=np.array([[1,2],[2,4]])

print(np.add(a,b))

m1=np.array([[7,3,2],[7,5,1],[5,5,5]])
m2=np.array([[7,3,2],[7,5,1],[5,5,5]])

#prodotto tra matrici
print(np.dot(m1,m2))

#somma righe o colonne matrice
m1.sum()
print(m1.sum(axis=1))
print(m1.sum(axis=0))

#media 
print(m1.mean())