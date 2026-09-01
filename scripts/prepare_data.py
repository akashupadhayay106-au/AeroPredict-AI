import os
import pandas as pd
from src.config import RUL_CAP, PROCESSED_DATA_DIR
from src.data_loader import load_dataset
from src.rul import calculate_rul, calculate_test_rul
from src.feature_engineering import preprocess_pipeline
import joblib

def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        print(f"Processing {subset}...")
        
        # Load
        train_df = load_dataset(subset, "train")
        test_df = load_dataset(subset, "test")
        rul_df = load_dataset(subset, "rul")
        
        if train_df.empty or test_df.empty or rul_df.empty:
            continue
            
        # Add RUL
        train_df = calculate_rul(train_df, cap=RUL_CAP)
        test_df = calculate_test_rul(test_df, rul_df)
        
        # Feature Engineering
        train_feat, test_feat, dropped_cols = preprocess_pipeline(train_df, test_df)
        print(f"Dropped constant columns: {dropped_cols}")
        
        # Save processed
        train_path = os.path.join(PROCESSED_DATA_DIR, f"train_{subset}_processed.csv")
        test_path = os.path.join(PROCESSED_DATA_DIR, f"test_{subset}_processed.csv")
        
        train_feat.to_csv(train_path, index=False)
        test_feat.to_csv(test_path, index=False)
        
        print(f"Saved {subset} to {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()
