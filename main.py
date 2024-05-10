from data_loader import *
from preprocessing import *

if __name__=="__main":
    filepath = ""
    data_content = load_data(filepath)
    preproc_data = scale_features(data_content)
    preproc_lables = encode_labels(data_content)

    #Call algorithm 1 - Logistic Regression


    #Call agorithm 2 - SVM