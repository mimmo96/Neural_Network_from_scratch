from Model_Selection import ThreadPool_average
import Ensemble
import File_names as fn

#return best K models
def Hold_out(epochs, grid, training_set, validation_set, test_set, type_problem,tot):
    num_training = 1
    top_k_models = Ensemble.Ensemble()
    
    for hyperparameter in grid:
        print("-----------------------------------START MODEL: ", num_training,"/",tot)
        #calculate the average of the 5 generated models and return the loss and the MEE / accuracy
        model_stat = ThreadPool_average(type_problem, training_set, validation_set, epochs, num_training, hyperparameter)

        #insert model_stat in best model if it is in K top model
        top_k_models.insert_model(model_stat)
        model_stat.write_result(fn.general_results)
        num_training += 1   
        print("----------------------------------------END-------------------------------")     
    
    #top_k_models contiene Ensemble di 10 migliori modelli e write_result stampa nel file csv tutti i miglioir 10 modelli e i loro parametri
    #Ensamble on 10 top models on test set
    top_k_models.write_result(fn.top_general_results)
    top_k_models.loss_average(test_set,fn.top_result_test)

    return top_k_models.getNN()
