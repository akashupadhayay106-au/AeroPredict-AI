import os
import pandas as pd
from src.train_ml import train_and_evaluate_ml
from src.train_dl import train_all_dl

def main():
    print("--- Starting ML Training ---")
    os.system("python -m src.train_ml")
    
    print("\n--- Starting DL Training ---")
    os.system("python -m src.train_dl")
    
    print("\n--- All Training Complete ---")
    print("Run python -m src.model_selection to compare and select the best model.")

if __name__ == "__main__":
    main()
