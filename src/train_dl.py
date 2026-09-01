import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import PROCESSED_DATA_DIR, SEQUENCE_LENGTH, DL_MODEL_DIR
from src.sequence_builder import generate_sequences, generate_test_sequences

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

class CNN1DModel(nn.Module):
    def __init__(self, input_dim, seq_len):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Calculate flattened dimension
        flatten_dim = 128 * (seq_len // 4)  # 2 pooling layers
        self.fc1 = nn.Linear(flatten_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # x is (batch, seq_len, features) -> Conv1d expects (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        out = self.fc2(x)
        return out

def train_model(model, train_loader, val_loader, epochs=15, lr=0.001, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch).squeeze()
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            
    model.load_state_dict(best_model_state)
    return model

def evaluate_dl(model, test_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_y = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).squeeze()
            all_preds.extend(preds.cpu().numpy())
            all_y.extend(y_batch.numpy())
            
    all_preds = np.array(all_preds)
    all_y = np.array(all_y)
    
    mae = mean_absolute_error(all_y, all_preds)
    rmse = np.sqrt(mean_squared_error(all_y, all_preds))
    r2 = r2_score(all_y, all_preds)
    return mae, rmse, r2

def train_all_dl():
    file_path_train = os.path.join(PROCESSED_DATA_DIR, "train_FD001_processed.csv")
    file_path_test = os.path.join(PROCESSED_DATA_DIR, "test_FD001_processed.csv")
    
    train_df = pd.read_csv(file_path_train)
    test_df = pd.read_csv(file_path_test)
    
    features = [c for c in train_df.columns if c not in ['unit_number', 'time_cycles', 'RUL', 'dataset_id']]
    
    # Generate sequences
    X_train, y_train = generate_sequences(train_df, features, 'RUL', SEQUENCE_LENGTH)
    X_test, y_test = generate_test_sequences(test_df, features, 'RUL', SEQUENCE_LENGTH)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    models = {
        'LSTM': LSTMModel(input_dim=len(features)).to(device),
        'GRU': GRUModel(input_dim=len(features)).to(device),
        '1D CNN': CNN1DModel(input_dim=len(features), seq_len=SEQUENCE_LENGTH).to(device)
    }
    
    results = []
    
    os.makedirs(DL_MODEL_DIR, exist_ok=True)
    
    for name, model in models.items():
        print(f"Training {name}...")
        model = train_model(model, train_loader, test_loader, epochs=15, device=device)
        mae, rmse, r2 = evaluate_dl(model, test_loader, device=device)
        
        results.append({'Model': name, 'Type': 'Deep Learning', 'MAE': mae, 'RMSE': rmse, 'R2': r2})
        torch.save(model.state_dict(), os.path.join(DL_MODEL_DIR, f"{name.lower().replace(' ', '_')}.pth"))
        
    results_df = pd.DataFrame(results)
    
    os.makedirs('reports/metrics', exist_ok=True)
    results_df.to_csv('reports/metrics/dl_comparison.csv', index=False)
    
    print("\nDL Evaluation Results:")
    print(results_df)
    
if __name__ == "__main__":
    train_all_dl()
