import numpy as np
from logistic_regression import train_logistic_regression, evaluate_model

def test_train_logistic_regression():
    # Dummy dataset
    X = np.random.rand(100, 4)  # 100 samples, 4 features
    y = np.random.randint(0, 2, size=(100,))  # Binary target

    # Train the model
    model, X_test, y_test = train_logistic_regression(X, y)
    assert model is not None, "Model training failed."
    assert len(X_test) == 20, "Test set size should be 20% of the total."
    assert len(y_test) == 20, "Test set size should be 20% of the total."

def test_evaluate_model():
    # Create a dummy dataset
    X = np.random.rand(20, 4)  # 20 samples, 4 features
    y = np.random.randint(0, 2, size=(20,))  # Binary target
    model, _, _ = train_logistic_regression(X, y)

    # Capture the output of the evaluation
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    with redirect_stdout(f):
        evaluate_model(model, X, y)
    out = f.getvalue()
    assert "precision" in out, "Evaluation output should contain 'precision'."
