from leggifile import print_result
import neural_network
import numpy as np
import ensemble
from function import LOSS, accuracy,MEE
from concurrent.futures import ThreadPoolExecutor

def model_selection(vector_alfa, vector_learning_rate, vector_lambda, vectors_units, training_set, validation_set,test_set, batch_array, epoch_array,fun,fun_output, weight,type_problem):
    
    #inizializzo i parametri che mi serviranno per salvare i dati dopo il training (per la scelta del miglior modello)
    best_learning_rate =vector_learning_rate[0]
    best_v_lambda=vector_lambda[0]
    best_units=vectors_units[0]
    best_alfa=vector_alfa[0]
    num_training=1;
    best_loss_validation=-1
    best_function=1
    MEE=0
    # vettore dove salvo i migliori 10 modelli
    best_NN=np.empty(0,neural_network.neural_network)
    # file dove andrò a scrivere 
    out_file = open("result/resultexecution.txt","w")
   
    for epochs in epoch_array:
        for batch_size in batch_array:
            for function in fun:
                for fun_out in fun_output:
                    for units in vectors_units:
                        for learning_rate in vector_learning_rate:
                            for alfa in vector_alfa:
                                for v_lambda in vector_lambda:
                                    for weig in weight:
                                        print_result(out_file,"------------------------------INIZIO TRAINING "+str(num_training)+"-----------------------------")
                                        print_result(out_file,"epoch:"+str(epochs)+" batch_size:"+str(batch_size)+" alfa:"+str(alfa)+ 
                                        "  lamda:"+ str(v_lambda)+ "  learning_rate:"+str(learning_rate) +
                                        "  layer:"+str(units)+ " function:"+str(function)+ " weight:"+ str(weig))
                                        #salvo il numero di layer 
                                        numero_layer=np.size(units)-2
                            
                                        #calcvolo la media dei 5 modelli generati e restituisco la loss e la MEE/accuratezza
                                        loss_validation,MEE,NN=ThreadPool_average(type_problem,fun_out,training_set,validation_set, batch_size, epochs,num_training,units, alfa, v_lambda, learning_rate, numero_layer,weig,function)
                                        #stampo il risultato di fine training
                                        print_result(out_file,"RESULT:") 
                                        print_result(out_file,"MEDIA LOSS:"+ str(loss_validation)+ " \nMEDIA MEE:"+ str(MEE))
    
                                        if best_loss_validation == -1:
                                            best_loss_validation = loss_validation
                                            best_model = NN
                                            best_NN=save(best_NN,NN)
                                        else:
                                            if loss_validation < best_loss_validation: 
                                                print_result(out_file,"aggiorno l'errore! \n vecchio errore:"+ str(best_loss_validation)+
                                                "\n nuovo errore:"+ str(loss_validation))
                                                best_loss_validation = loss_validation
                                                best_model = NN
                                                best_NN=save(best_NN,best_model)
                                                best_learning_rate =learning_rate
                                                best_v_lambda=v_lambda
                                                best_units=units
                                                best_alfa=alfa
                                                best_function=function
                                                #salvo il nuovo modello nell'array

                                        print_result(out_file,"------------------------------FINE TRAINING "+str(num_training)+"-----------------------------")
                                        num_training=num_training+1        

    print_result(out_file,"\n************************* RESULT **************************")
    print_result(out_file,"parametri migliori:  epoch:"+str(epochs)+" batch_size:"+str(batch_size)+" alfa:"+str(best_alfa)+ "  lamda:"+ str(best_v_lambda)+ "  learning_rate:"
                +str(best_learning_rate) +"  layer:"+str(best_units)+ " function:"+str(best_function)+ " weight:"+str(weig))
    print_result(out_file,"\nerrore sul validation migliore: "+str(best_loss_validation)) 

    #sui 10 migliori modelli trovati mi salvo i risultati
    file = open("result/best_model/result.txt","w")
    conta=1
    for neural in best_NN:
        output_NN = np.zeros(test_set.output().shape)
        neural.ThreadPool_Forward(test_set.input(), 0, test_set.input().shape[0], output_NN, True)
        loss_test = LOSS(output_NN, test_set.output(), test_set.output().shape[0],0)
        print_result(file,"------------------------------MODEL "+str(conta)+"-----------------------------")
        print_result(file,"parametri:  alfa:"+str(neural.alfa)+ "  lamda:"+ str(neural.v_lambda)+ "  learning_rate:"
                +str(neural.learning_rate) +"  layer:"+str(neural.nj)+ " function:"+str(neural.function)+ " weight:"+str(neural.type_weight))
 
        print_result(file,"RESULT on TEST SET "+str(loss_test))
        print_result(file,"--------------------------------------------------------------------------------")
        conta=conta+1
   
    
    #faccio ensamble sul test set
    en=ensemble.ensemble(best_NN,test_set)
    loss_ensemble=en.loss_average()
    print_result(file,"errore sui test:" +str(loss_ensemble)) 
    
    #chiudo i file
    file.close()
    out_file.close()
    return best_model

# funzioni per parallelizzare il calcolo con i 5 modelli in modo da fare la media finale
def task(type_problem,fun_out, NN,training_set,validation_set, batch_size, epochs,num_training):
    NN.trainig(training_set, validation_set, batch_size, epochs,num_training) 
    output_NN = np.zeros(validation_set.output().shape)
    NN.ThreadPool_Forward(validation_set.input(), 0, validation_set.input().shape[0], output_NN, True)
    penalty_term = NN.penalty_NN()
    loss_tot = LOSS(output_NN, validation_set.output(), validation_set.output().shape[0], penalty_term)
    if(type_problem=="classification"):
        MEE_tot=accuracy(type_problem,fun_out,validation_set.output(),output_NN)
    else:
        MEE_tot=MEE(output_NN, validation_set.output(), validation_set.output().shape[0])
    
    return loss_tot,MEE_tot,output_NN
    

def ThreadPool_average(type_problem,fun_out,training_set,validation_set, batch_size, epochs,num_training,units, alfa, v_lambda, learning_rate, numero_layer,weig,function):
    #creo il pool di thread
    executor = ThreadPoolExecutor(5)
    loss_tot=0
    best_loss=100000000000000000
    MEE_tot=0
    best_NN=0

    for i in range(0,5):
        #creo la neural network con i parametri passati
        NN = neural_network.neural_network(units, alfa, v_lambda, learning_rate, numero_layer,function, fun_out, weig, type_problem) 
        loss,MEE,out=executor.submit(task,type_problem,fun_out, NN,training_set,validation_set, batch_size, epochs,num_training).result()
        print("loss ",i,":" ,loss)
        print("MEE ",i,":", MEE)
        if(loss<best_loss):
            best_loss=loss
            best_NN=NN
        best_loss=loss
        loss_tot = loss_tot + loss
        MEE_tot= MEE_tot +MEE
    
    executor.shutdown(True)

    loss_tot=np.divide(loss_tot,5)
    MEE_tot=np.divide(MEE_tot,5)

    return loss_tot,MEE_tot,best_NN


# top_model=array contente i migliori 10 modelli
# new_model= modello da inserire tra i 10
# ogni migliori modello verrà inserito nella prima posizione e farà scorrerre gli altri, se supero la dimensione di 10 cancello l'ultimo inserito 
def save(top_model,new_model):
    top_model=np.insert(top_model,0,new_model)
    if(np.size(top_model)>10):
        top_model=np.delete(top_model,-1)
    
    return top_model
