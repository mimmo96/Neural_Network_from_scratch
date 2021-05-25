from Model_Selection import ThreadPool_average
import Ensemble
import File_names as fn

#return best K models
def Hold_out(epochs, grid, training_set, validation_set, test_set, type_problem,tot,num_training):
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
    
    #top_k_models contains Ensemble of 8 best models and write_result prints in the csv file all the best 8 models and their parameters
    top_k_models.write_result(fn.top_general_results)
    #Ensamble on 8 top models on test set
    top_k_models.loss_average(test_set,fn.top_result_test)

    return top_k_models.getNN(),num_training
