from sklearn.preprocessing import StandardScaler, LabelEncoder

def scale_features(X):
    """
    Feature scaling using StandardScaler.
    """
    scaler=StandardScaler()
    X_scaled= scaler.fit_transform(X)
    return X_scaled

def encode_labels(y):
    """
    Encode labels using LabelEncoder.
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded
