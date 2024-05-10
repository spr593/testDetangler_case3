from sklearn.metrics import precision_score, recall_score, f1_score
from data_loader import *
from preprocessing import *
from logistic_regression import *
from svm import * 

def compare_models (ypred_m1, ypred_m2):
    precision = precision_score(ypred_m1, ypred_m2, average="binary")
    recall =  recall_score(ypred_m1, ypred_m2, average="binary")
    f1 = f1_score(ypred_m1, ypred_m2, average="binary")
    return precision, recall, f1


if __name__=="__main":
    filepath = ""
    X,y = load_data(filepath)
    preproc_data = scale_features(X,y)
    preproc_lables = encode_labels(X,y)

    #Call algorithm 1 - Logistic Regression


    #Call agorithm 2 - SVM
    svm = SVM(kernel='radial')
    Xtest_svm, ytest_svm = svm.train_svm (X,y)
    report = svm.evaluate_model(ytest_svm, y)
    
    #Compare models metrics