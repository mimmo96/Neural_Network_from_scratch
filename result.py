import pandas
import neural_network

def write_result(tuples_NN, file_csv):
    for tuple_NN in tuples_NN:
        NN = tuple_NN[0]
        error = tuple_NN[1]
        number_model = tuple_NN[-1]
        row_csv = {
            'Number_Model' : number_model,
            'Units_per_layer' : NN.nj,
            'learning_rate' : NN.learning_rate,
            'lambda' : NN.v_lambda,
            'alfa' : NN.alfa,
            'function_hidden' : NN.function,
            'inizialization_weights' : NN.type_weight,
            'Error_MSE' : error 
        }
        df = pandas.DataFrame(row_csv)
        df.to_csv(file_csv)

#def write_blind_test(output, file_txt):

    
