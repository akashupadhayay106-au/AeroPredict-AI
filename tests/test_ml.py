import pandas as pd
import numpy as np
from src.rul import calculate_rul
from src.feature_engineering import add_rolling_features

def test_calculate_rul():
    # Mock data for one engine
    df = pd.DataFrame({
        'unit_number': [1, 1, 1, 1],
        'time_cycles': [1, 2, 3, 4]
    })
    
    # Calculate RUL without cap
    result_df = calculate_rul(df)
    
    # Assert RUL calculation is correct (Max cycle is 4, RUL = 4 - cycle)
    expected_rul = [3, 2, 1, 0]
    np.testing.assert_array_equal(result_df['RUL'].values, expected_rul)
    
def test_calculate_rul_with_cap():
    df = pd.DataFrame({
        'unit_number': [1, 1, 1, 1],
        'time_cycles': [1, 2, 3, 4]
    })
    
    # Calculate RUL with cap = 2
    result_df = calculate_rul(df, cap=2)
    expected_rul = [2, 2, 1, 0]
    np.testing.assert_array_equal(result_df['RUL'].values, expected_rul)

def test_feature_engineering_no_leakage():
    # Create two engines
    df = pd.DataFrame({
        'unit_number': [1, 1, 2, 2],
        'time_cycles': [1, 2, 1, 2],
        'sensor_1': [10, 20, 100, 200]
    })
    
    # We patch SENSOR_COLUMNS for the test
    import src.config
    src.config.SENSOR_COLUMNS = ['sensor_1']
    
    # Add rolling features (window = 2)
    result = add_rolling_features(df, window_sizes=[2])
    
    # Check engine 1 rolling mean for cycle 2 -> (10+20)/2 = 15
    assert result.loc[1, 'sensor_1_roll_mean_2'] == 15.0
    
    # Check engine 2 rolling mean for cycle 1 -> (100)/1 = 100 
    # If leakage existed, it might have included engine 1's cycle 2 (20+100)/2 = 60
    assert result.loc[2, 'sensor_1_roll_mean_2'] == 100.0
