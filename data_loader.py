import pandas as pd

def load_data(filepath):
    """
    Load data from CSV file
    """
    return pd.read_csv(filepath)