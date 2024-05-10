import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_logistic_regression(X, y):
    """
    Train a logistic regression model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """ Evaluate the logistic regression model.
    """
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))