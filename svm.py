import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

class SVM:
    def __init__(self, kernel= 'linear'):
        self.kernel = kernel
        self.model = SVC(kernel=self.kernel)

    def train_svm(X, y):
        """Train a Support Vector Machine classifier"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = SVC(kernel='linear')
        model.fit(X_train, y_train)
        return model, X_test, y_test

    def evaluate_model(model, X_test, y_test):
        """Evaluate the SVM model."""
        predictions = model.predict(X_test)
        return classification_report(y_test, predictions)

    def test_evaluate_svm(self):
        """Test evaluation of SVM."""
        X = np.random.rand(20, 4)
        y = np.random.randint(0, 2, size=20)
        model, _, _ = self.train_svm(X, y)
        report = self.evaluate_model(model, X, y)
        assert 'precision' in report, "Evaluation should include precision"