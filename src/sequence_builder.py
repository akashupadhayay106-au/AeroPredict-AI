import numpy as np
import pandas as pd

def generate_sequences(df: pd.DataFrame, features: list, target: str, sequence_length: int = 30):
    """
    Generate 3D sequences (samples, sequence_length, features) from a DataFrame.
    Groups by unit_number to prevent crossing engine boundaries.
    """
    X_list = []
    y_list = []
    
    for unit_id, group in df.groupby('unit_number'):
        # Only take engines with enough data
        if len(group) >= sequence_length:
            # We want to create rolling windows
            features_data = group[features].values
            target_data = group[target].values
            
            for i in range(len(group) - sequence_length + 1):
                X_list.append(features_data[i : i + sequence_length])
                # Target is the RUL at the END of the sequence
                y_list.append(target_data[i + sequence_length - 1])
                
    if len(X_list) == 0:
        return np.array([]), np.array([])
        
    return np.array(X_list), np.array(y_list)

def generate_test_sequences(df: pd.DataFrame, features: list, target: str, sequence_length: int = 30):
    """
    For the test set, we only care about predicting the LAST sequence of each engine,
    since the RUL provided in the test labels corresponds to the end of the observed sequence.
    """
    X_list = []
    y_list = []
    
    for unit_id, group in df.groupby('unit_number'):
        if len(group) >= sequence_length:
            features_data = group[features].values
            target_data = group[target].values
            
            # Take only the last sequence
            X_list.append(features_data[-sequence_length:])
            y_list.append(target_data[-1])
            
    if len(X_list) == 0:
        return np.array([]), np.array([])
        
    return np.array(X_list), np.array(y_list)
