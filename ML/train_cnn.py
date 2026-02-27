import os
import sys
# Ensure project root is in path regardless of CWD
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from tqdm import tqdm
from ML.data_loader import SpatialDataLoader

OUT_DIR = os.path.join(_PROJECT_ROOT, "ML/results/cnn")
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = ["ph", "soc", "no3", "p", "k", "s"]

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n" + "="*54)
print(f"  PyTorch : {torch.__version__}")
if torch.cuda.is_available():
    dev_name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  Device  : CUDA — {dev_name} ({mem_gb:.1f} GB VRAM)")
else:
    import os
    print(f"  Device  : CPU — {os.cpu_count()} cores (no CUDA available)")
    print("  Hint    : install torch with CUDA support for faster training")
print(f"  Running : {DEVICE}")
print("="*54 + "\n")

# Hyperparameters
EPOCHS = 300
BATCH_SIZE = 32
PATIENCE = 30
LR = 0.001

class SoilCNN1D(nn.Module):
    """
    1D Convolutional Neural Network for Soil Prediction based on 15 features.
    Input shape: (batch_size, channels=1, sequence_length=15)
    """
    def __init__(self, input_size=15):
        super(SoilCNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        
        # Calculate flattened size
        # L_out = ((L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1
        # pool 1: size = floor(15 / 2) = 7
        # pool 2: size = floor(7 / 2) = 3
        # flat_size = 32 * 3 = 96
        self.flatten_size = 96
        
        self.fc1 = nn.Linear(self.flatten_size, 32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch, 1, seq_len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.dropout(x)
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

def train_and_evaluate(target: str):
    print(f"\n{'='*50}")
    print(f" Training 1D CNN for: {target.upper()}")
    print(f"{'='*50}")
    
    loader = SpatialDataLoader(target=target, scale_features=True) # CNN needs scaling
    X, y, fields = loader.get_data()
    feature_names = loader.get_feature_names()
    
    print(f"Data shape: {X.shape}, Unique fields: {len(np.unique(fields))}")
    print(f"Features used: {len(feature_names)}")
    
    oof_preds = np.zeros_like(y)
    
    lofo_iter = list(loader.iter_lofo_cv())
    n_folds = len(lofo_iter)

    fold_train_losses = []
    fold_val_losses = []

    fold_bar = tqdm(lofo_iter, desc=f"  Folds [{target.upper()}]",
                    unit="fold", ncols=90, colour='cyan')

    for fold, (train_idx, test_idx, test_field) in enumerate(fold_bar):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Make another split for Early Stopping inside the train set (LOFO on train)
        # We will split 20% of the train fields for true validation, avoiding leakage.
        unique_train_fields = np.unique(fields[train_idx])
        np.random.seed(42 + fold)
        val_fields = np.random.choice(unique_train_fields, size=int(len(unique_train_fields) * 0.2), replace=False)
        
        val_mask = np.isin(fields[train_idx], val_fields)
        tr_mask = ~val_mask
        
        X_tr, y_tr = X_train[tr_mask], y_train[tr_mask]
        X_val, y_val = X_train[val_mask], y_train[val_mask]
        
        # Convert to Tensors
        # CNN expects (batch_size, channels, sequence_length)
        tensor_X_tr = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        tensor_y_tr = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1).to(DEVICE)
        
        tensor_X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        tensor_y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(DEVICE)
        
        tensor_X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        
        # DataLoaders
        train_dataset = TensorDataset(tensor_X_tr, tensor_y_tr)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        model = SoilCNN1D(input_size=len(feature_names)).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4) # L2 regularization
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        train_losses = []
        val_losses = []
        
        # Training Loop with tqdm epoch bar
        epoch_bar = tqdm(range(EPOCHS), desc=f"    Epochs", leave=False,
                         unit="ep", ncols=90, colour='green')

        for epoch in epoch_bar:
            model.train()
            epoch_loss = 0.0
            for b_x, b_y in train_loader:
                b_x = b_x.to(DEVICE)
                b_y = b_y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(b_x)
                loss = criterion(outputs, b_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(tensor_X_val)
                val_loss = criterion(val_outputs, tensor_y_val).item()
                val_losses.append(val_loss)
                
            epoch_bar.set_postfix(train=f"{avg_train_loss:.4f}",
                                  val=f"{val_loss:.4f}",
                                  patience=patience_counter)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                
            if patience_counter >= PATIENCE:
                epoch_bar.set_postfix(stopped=f"epoch {epoch}",
                                      val=f"{val_loss:.4f}")
                epoch_bar.close()
                break
                
        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)
        
        # Load best model and predict on test fold
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        model.eval()
        with torch.no_grad():
            test_preds = model(tensor_X_test).cpu().numpy().flatten()
            
        oof_preds[test_idx] = test_preds
        fold_bar.set_postfix(field=test_field, val_loss=f"{best_val_loss:.4f}")
        
    # Validation Overall Metrics
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    mae = mean_absolute_error(y, oof_preds)
    r2 = r2_score(y, oof_preds)
    
    rho, p_val = spearmanr(y, oof_preds)
    
    print(f"\n[Results {target.upper()}]")
    print(f"  Spearman rho: {rho:.3f} (p={p_val:.2e})")
    print(f"  RMSE:         {rmse:.3f}")
    print(f"  MAE:          {mae:.3f}")
    print(f"  R2:           {r2:.3f}")
    
    # Save OOF
    oof_df = loader.df.copy()
    oof_df['oof_pred'] = oof_preds
    oof_df.to_csv(os.path.join(OUT_DIR, f"{target}_oof_predictions.csv"), index=False)
    
    # Plot Scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(y, oof_preds, alpha=0.5, edgecolor='k')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.title(f"1D CNN LOFO CV - {target.upper()}\nSpearman $\\rho$ = {rho:.3f}")
    plt.xlabel(f"True {target.upper()}")
    plt.ylabel(f"Predicted {target.upper()}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{target}_scatter.png"), dpi=300)
    plt.close()
    
    # Plot Learning Curves for Fold 0
    plt.figure(figsize=(8, 5))
    plt.plot(fold_train_losses[0], label='Train Loss')
    plt.plot(fold_val_losses[0], label='Val Loss')
    plt.title(f"CNN Learning Curve (Fold 0) - {target.upper()}")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, f"{target}_learning_curve_f0.png"), dpi=300)
    plt.close()
    
    return {
        "rho": float(rho),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }

if __name__ == "__main__":
    results_all = {}
    for t in TARGETS:
        try:
            metrics = train_and_evaluate(t)
            results_all[t] = metrics
        except Exception as e:
            print(f"Error training {t}: {e}")
            
    with open(os.path.join(OUT_DIR, "cnn_metrics_summary.json"), "w") as f:
        json.dump(results_all, f, indent=4)
        
    print(f"\n[DONE] All targets processed. Results saved in {OUT_DIR}/")
