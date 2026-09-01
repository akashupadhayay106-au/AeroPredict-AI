import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import PROCESSED_DATA_DIR, MODEL_DIR

def train_and_evaluate_ml(train_df, test_df, features, target='RUL'):
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1),
        'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1)
    }
    
    results = []
    best_r2 = -float('inf')
    best_model_name = None
    best_model = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        results.append({'Model': name, 'Type': 'Classical ML', 'MAE': mae, 'RMSE': rmse, 'R2': r2})
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = model
            
    # Save the best model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, f"best_ml_model.pkl"))
    
    # Save feature names for inference
    joblib.dump(features, os.path.join(MODEL_DIR, "feature_config.pkl"))
    
    return pd.DataFrame(results), best_model_name

if __name__ == "__main__":
    file_path_train = os.path.join(PROCESSED_DATA_DIR, "train_FD001_processed.csv")
    file_path_test = os.path.join(PROCESSED_DATA_DIR, "test_FD001_processed.csv")
    
    train_df = pd.read_csv(file_path_train)
    test_df = pd.read_csv(file_path_test)
    
    # Drop non-feature columns
    features = [c for c in train_df.columns if c not in ['unit_number', 'time_cycles', 'RUL', 'dataset_id']]
    
    results_df, best_model = train_and_evaluate_ml(train_df, test_df, features)
    
    os.makedirs('reports/metrics', exist_ok=True)
    results_df.to_csv('reports/metrics/ml_comparison.csv', index=False)
    
    print("\nML Evaluation Results:")
    print(results_df)
    print(f"\nBest ML Model Saved: {best_model}")
