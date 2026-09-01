import os
import pandas as pd
from src.config import RAW_DATA_DIR, INDEX_COLUMNS, SETTING_COLUMNS, SENSOR_COLUMNS

def load_dataset(subset: str = "FD001", data_type: str = "train") -> pd.DataFrame:
    """
    Load specific subset of C-MAPSS data.
    
    Args:
        subset: 'FD001', 'FD002', 'FD003', or 'FD004'
        data_type: 'train', 'test', or 'rul'
    """
    if data_type == "rul":
        file_path = os.path.join(RAW_DATA_DIR, "rul", f"RUL_{subset}.csv")
        # RUL files typically don't have headers, or if they do, we can check.
        # But based on standard CMAPSS, it's just a single column of max RULs for the test set.
        try:
            df = pd.read_csv(file_path, header=0, names=["max_rul"])
            df['unit_number'] = df.index + 1
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return pd.DataFrame()
            
    file_path = os.path.join(RAW_DATA_DIR, data_type, f"{data_type}_{subset}.csv")
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def get_all_subsets(data_type: str = "train") -> pd.DataFrame:
    """Load and concatenate all available subsets."""
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    df_list = []
    
    for subset in subsets:
        df = load_dataset(subset, data_type)
        if not df.empty:
            df['dataset_id'] = subset
            df_list.append(df)
            
    if not df_list:
        raise ValueError(f"No {data_type} data found.")
        
    return pd.concat(df_list, ignore_index=True)
