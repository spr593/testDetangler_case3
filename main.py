from data_loader import *
from preprocessing import *
from sklearn.metrics import precision_score, recall_score, f1_score

def compare_models (ypred_m1, ypred_m2):
    precision = precision_score(ypred_m1, ypred_m2, average="binary")
    recall =  recall_score(ypred_m1, ypred_m2, average="binary")
    f1 = f1_score(ypred_m1, ypred_m2, average="binary")
    return precision, recall, f1


if __name__=="__main":
    filepath = ""
    data_content = load_data(filepath)
    preproc_data = scale_features(data_content)
    preproc_lables = encode_labels(data_content)

    #Call algorithm 1 - Logistic Regression


    #Call agorithm 2 - SVM


    #Compare models metrics