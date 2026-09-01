import pandas as pd
from src.config import SENSOR_COLUMNS

def add_rolling_features(df: pd.DataFrame, window_sizes: list = [5, 10]) -> pd.DataFrame:
    """
    Add rolling mean and standard deviation for sensor columns.
    Grouping by unit_number prevents leaking data between engines.
    """
    df_out = df.copy()
    
    valid_sensors = [col for col in SENSOR_COLUMNS if col in df_out.columns]
    if not valid_sensors:
        return df_out
    
    for w in window_sizes:
        # Rolling Mean
        rolling_mean = df_out.groupby('unit_number')[valid_sensors].rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
        rolling_mean.columns = [f"{col}_roll_mean_{w}" for col in valid_sensors]
        
        # Rolling Std
        rolling_std = df_out.groupby('unit_number')[valid_sensors].rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True)
        rolling_std.columns = [f"{col}_roll_std_{w}" for col in valid_sensors]
        # Fill NaN for the first element where std is undefined
        rolling_std.fillna(0, inplace=True)
        
        df_out = pd.concat([df_out, rolling_mean, rolling_std], axis=1)
        
    return df_out

def preprocess_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Applies the full feature engineering pipeline.
    Ensures no leakage from test to train.
    """
    train_feat = add_rolling_features(train_df)
    test_feat = add_rolling_features(test_df)
    
    # We might want to drop highly correlated or constant sensors.
    # Usually in C-MAPSS FD001, sensors 1, 5, 10, 16, 18, 19 are constant.
    # Let's drop constant columns dynamically based on train_df
    
    constant_cols = [col for col in SENSOR_COLUMNS if train_feat[col].std() < 1e-4]
    
    # Also drop from test
    train_feat.drop(columns=constant_cols, inplace=True, errors='ignore')
    test_feat.drop(columns=constant_cols, inplace=True, errors='ignore')
    
    return train_feat, test_feat, constant_cols
