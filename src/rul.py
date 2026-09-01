import pandas as pd
import numpy as np

def calculate_rul(df: pd.DataFrame, cap: int = None) -> pd.DataFrame:
    """
    Calculates the Remaining Useful Life (RUL) for each row in the training dataset.
    
    Args:
        df: DataFrame containing at least 'unit_number' and 'time_cycles'
        cap: Optional integer to cap the maximum RUL (e.g., 125 or 130).
             Capping helps models focus on the degradation phase rather than 
             early healthy phases where sensor readings are stable.
             
    Returns:
        DataFrame with a new 'RUL' column.
    """
    # Find the maximum cycle for each unit
    max_cycles = df.groupby('unit_number')['time_cycles'].max().reset_index()
    max_cycles.rename(columns={'time_cycles': 'max_cycle'}, inplace=True)
    
    # Merge back to original dataframe
    df = df.merge(max_cycles, on=['unit_number'], how='left')
    
    # Calculate RUL
    df['RUL'] = df['max_cycle'] - df['time_cycles']
    
    # Drop intermediate column
    df.drop(columns=['max_cycle'], inplace=True)
    
    if cap is not None:
        df['RUL'] = np.minimum(df['RUL'], cap)
        
    return df

def calculate_test_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the RUL for the test dataset.
    In C-MAPSS, test engines don't run to failure. The rul_df contains the true RUL 
    for the LAST recorded cycle of each engine in the test set.
    """
    # Get the max cycle for each engine in the test set
    max_cycles = test_df.groupby('unit_number')['time_cycles'].max().reset_index()
    max_cycles.rename(columns={'time_cycles': 'max_cycle'}, inplace=True)
    
    # Merge true RUL at the end of the test sequence
    max_cycles = max_cycles.merge(rul_df, on='unit_number', how='left')
    
    # The true maximum cycle if the engine had run to failure would be:
    # current_max_cycle + true_remaining_rul
    max_cycles['absolute_max_cycle'] = max_cycles['max_cycle'] + max_cycles['max_rul']
    
    # Merge back
    test_df = test_df.merge(max_cycles[['unit_number', 'absolute_max_cycle']], on='unit_number', how='left')
    
    test_df['RUL'] = test_df['absolute_max_cycle'] - test_df['time_cycles']
    test_df.drop(columns=['absolute_max_cycle'], inplace=True)
    
    return test_df
