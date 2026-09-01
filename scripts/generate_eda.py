import os
import nbformat as nbf

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    
    text = """\
# AeroPredict AI - Exploratory Data Analysis (EDA)
This notebook performs EDA on the NASA C-MAPSS dataset.
We will inspect sensor distributions, correlations, and engine degradation.
"""
    code_1 = """\
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create reports dir if not exists
os.makedirs('../reports/figures', exist_ok=True)

from src.data_loader import load_dataset
from src.rul import calculate_rul

# Load FD001 Train
df = load_dataset('FD001', 'train')
df = calculate_rul(df)
df.head()
"""
    code_2 = """\
# Dataset Overview
print(f"Total Engines: {df['unit_number'].nunique()}")
print(f"Total Cycles: {len(df)}")
"""
    code_3 = """\
# Sensor Distribution (Boxplots)
sensor_cols = [col for col in df.columns if 'sensor_' in col]
plt.figure(figsize=(15, 8))
sns.boxplot(data=df[sensor_cols])
plt.xticks(rotation=90)
plt.title('Sensor Distributions')
plt.savefig('../reports/figures/sensor_distribution.png')
plt.show()
"""
    code_4 = """\
# Sensor Correlation
plt.figure(figsize=(12, 10))
corr = df[sensor_cols].corr()
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title('Sensor Correlation Heatmap')
plt.savefig('../reports/figures/sensor_correlation.png')
plt.show()
"""
    code_5 = """\
# Degradation over time for Engine 1
engine_1 = df[df['unit_number'] == 1]
plt.figure(figsize=(10, 5))
plt.plot(engine_1['time_cycles'], engine_1['sensor_11'], label='Sensor 11')
plt.plot(engine_1['time_cycles'], engine_1['sensor_4'], label='Sensor 4')
plt.xlabel('Cycle')
plt.ylabel('Sensor Value')
plt.title('Engine 1 Sensor Degradation over time')
plt.legend()
plt.savefig('../reports/figures/engine1_degradation.png')
plt.show()
"""
    
    nb['cells'] = [
        nbf.v4.new_markdown_cell(text),
        nbf.v4.new_code_cell(code_1),
        nbf.v4.new_code_cell(code_2),
        nbf.v4.new_code_cell(code_3),
        nbf.v4.new_code_cell(code_4),
        nbf.v4.new_code_cell(code_5)
    ]
    
    with open('notebooks/02_eda.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_eda_notebook()
    print("EDA Notebook created at notebooks/02_eda.ipynb")
