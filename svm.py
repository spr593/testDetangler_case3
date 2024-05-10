from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def train_svm(X, y):
    """Train a Support Vector Machine classifier"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model, X_test, y_test