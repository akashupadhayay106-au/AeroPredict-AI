import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.config import PROCESSED_DATA_DIR

def run_eda():
    os.makedirs('reports/figures', exist_ok=True)
    
    file_path = os.path.join(PROCESSED_DATA_DIR, "train_FD001_processed.csv")
    if not os.path.exists(file_path):
        print(f"{file_path} not found. Run prepare_data.py first.")
        return
        
    df = pd.read_csv(file_path)
    
    print(f"Total Engines: {df['unit_number'].nunique()}")
    print(f"Total Cycles: {len(df)}")
    
    # 1. Sensor Distributions
    sensor_cols = [c for c in df.columns if 'sensor_' in c and 'roll' not in c]
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df[sensor_cols])
    plt.xticks(rotation=90)
    plt.title('Sensor Distributions')
    plt.tight_layout()
    plt.savefig('reports/figures/sensor_distribution.png')
    plt.close()
    
    # 2. Sensor Correlation
    plt.figure(figsize=(12, 10))
    corr = df[sensor_cols].corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title('Sensor Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('reports/figures/sensor_correlation.png')
    plt.close()
    
    # 3. Degradation over time for Engine 1
    engine_1 = df[df['unit_number'] == 1]
    plt.figure(figsize=(10, 5))
    if 'sensor_11' in df.columns:
        plt.plot(engine_1['time_cycles'], engine_1['sensor_11'], label='Sensor 11')
    if 'sensor_4' in df.columns:
        plt.plot(engine_1['time_cycles'], engine_1['sensor_4'], label='Sensor 4')
    plt.xlabel('Cycle')
    plt.ylabel('Sensor Value')
    plt.title('Engine 1 Sensor Degradation over time')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/figures/engine1_degradation.png')
    plt.close()

if __name__ == "__main__":
    run_eda()
    print("EDA Figures generated in reports/figures/")
