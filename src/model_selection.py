import os
import pandas as pd

def select_best_model():
    ml_metrics_path = 'reports/metrics/ml_comparison.csv'
    dl_metrics_path = 'reports/metrics/dl_comparison.csv'
    
    if os.path.exists(ml_metrics_path):
        ml_df = pd.read_csv(ml_metrics_path)
    else:
        ml_df = pd.DataFrame()
        
    if os.path.exists(dl_metrics_path):
        dl_df = pd.read_csv(dl_metrics_path)
    else:
        dl_df = pd.DataFrame()
        
    if ml_df.empty and dl_df.empty:
        print("No evaluation metrics found.")
        return
        
    combined = pd.concat([ml_df, dl_df], ignore_index=True)
    combined.to_csv('reports/model_comparison.csv', index=False)
    
    print("\n--- Overall Model Comparison ---")
    print(combined.sort_values(by='RMSE'))
    
    best_model_row = combined.loc[combined['RMSE'].idxmin()]
    
    print(f"\nBest Model Selected: {best_model_row['Model']} ({best_model_row['Type']})")
    print(f"Metrics - RMSE: {best_model_row['RMSE']:.4f}, MAE: {best_model_row['MAE']:.4f}, R2: {best_model_row['R2']:.4f}")
    
    with open('reports/metadata/best_model_info.txt', 'w') as f:
        f.write(f"Model: {best_model_row['Model']}\n")
        f.write(f"Type: {best_model_row['Type']}\n")
        f.write(f"RMSE: {best_model_row['RMSE']:.4f}\n")
        f.write(f"MAE: {best_model_row['MAE']:.4f}\n")
        f.write(f"R2: {best_model_row['R2']:.4f}\n")

if __name__ == "__main__":
    os.makedirs('reports/metadata', exist_ok=True)
    select_best_model()
