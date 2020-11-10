print ('ciao a tutti')
print ('il mio primo codice phyon!')
print ('ciao a tutti')

#variabili
age= 35
nome="rob"
asss="ciao io sono gianni"

print (nome,age )
print (asss[0:10])

# array
array=["Rob","Kirstin","Tommy"]
print(array[1])

# dizionari
dict= {}

dict["dad"]="Rob"
dict["mum"]="kirsten"
dict[1] = "tommy"
dict [2] ="Ralphie"

print (dict)
print (dict["mum"])
print (dict.keys())

# i cicli  
for i in range(5,11):
    print (i)

cibipreferiti=["pizza","CIOCCOLATO","gelato"]

for food in cibipreferiti:
    print("mi piace " + food + ".")

x=0
while x<=10:
    print (x)
    x+=1

#esercizio
nomi={}
nomi["angelo"]=64;
nomi["francesca"]=67;
nomi["sonia"]=21;
nomi["clelia"]=16;

for i in nomi:
    print(i,str(nomi[i]))

#condizioni
nome="mimmo"

if nome=="mimmo": 
    print("ciao mimmo!")

#funzioni 

def stampami(provadistampa):
    print (provadistampa)

def molt(x,y):
    return x*y

stampami("ci siamo")
print(molt(2,3))

