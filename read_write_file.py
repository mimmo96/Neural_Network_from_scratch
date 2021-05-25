import pandas


#####################
## READING METHODS ##
#####################


def read_csv(type_problem,file_name):
    if (type_problem == "Regression"):   
        file_csv = pandas.read_csv(file_name, delimiter = ',', 
                                    names = ["data", "x_1", "x_2","x_3", "x_4","x_5", "x_6","x_7", "x_8","x_9", "x_10", "label_1", "label_2"], 
                                    skiprows= 7)

        file_csv = file_csv.drop(["data"], axis = 1)
        
        matrix = file_csv.to_numpy()

    elif (type_problem == "classification"):
        
        file_csv = pandas.read_csv(file_name, delimiter = ' ', names = ["label", "x_1", "x_2","x_3", "x_4","x_5", "x_6", "data"])
        
        file_csv = file_csv.reindex(columns=[ "x_1", "x_2","x_3", "x_4","x_5", "x_6","label"])
        
        matrix = file_csv.to_numpy()

    elif (type_problem == "blind_test"):

        file_csv = pandas.read_csv(file_name, delimiter = ',', 
                                    names = ["data", "x_1", "x_2","x_3", "x_4","x_5", "x_6","x_7", "x_8","x_9", "x_10"], 
                                    skiprows= 7)

        file_csv = file_csv.drop(["data"], axis = 1)
        
        matrix = file_csv.to_numpy()
    #add new cases below..

    return matrix
