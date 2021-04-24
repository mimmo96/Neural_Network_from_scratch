import pandas
import numpy


def sort ():
    databasefile = "result/all_models.csv"
    db = pandas.read_csv(databasefile, delimiter =',')
    db.drop_duplicates(inplace=True)
    db = db.sort_values(by=["Error_MSE_vl"], ascending=True)
    db.to_csv("result/sorted_monk_1.csv", index = False, sep = ";")
    print(db)
sort()
