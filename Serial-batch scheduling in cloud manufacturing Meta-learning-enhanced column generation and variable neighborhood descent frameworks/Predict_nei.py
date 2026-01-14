import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch, torch.nn as nn, learn2learn as l2l
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import joblib
from pathlib import Path
from ast import literal_eval

MODEL_PATH = Path("maml_neighbor_model.joblib")  
df = pd.read_csv("test_CM\\result_VND.csv")
mse_hist, rmse_hist = [], []

def extract_features(row: dict, neighbor_idx: int):
    for col in ['r_j', 't_k', 'p_j', 'L_k', 'S_k', 'CP_k']:
        val = row[col]
        if isinstance(val, str):
            row[col] = literal_eval(val)

    scalar = [row['N'], row['T'], row['seta'], row['c'], neighbor_idx]
    seq_stats = []
    for col in ['r_j', 't_k', 'p_j', 'L_k', 'S_k', 'CP_k']:
        v = row[col]
        seq_stats += [np.mean(v), np.std(v), np.max(v), np.min(v)]
    return np.array(scalar + seq_stats, dtype=np.float32)

class NeighborTargetDataset(Dataset):
    @staticmethod
    def _safe_literal_eval(x):
        if isinstance(x, (list, np.ndarray)):
            return x
        if isinstance(x, str):
            try:
                return literal_eval(x)
            except Exception:
                return x
        return x

    def __init__(self, df: pd.DataFrame):
        df = df.copy()

        for col in ['r_j', 't_k', 'p_j', 'L_k', 'S_k', 'CP_k']:
            df[col] = df[col].apply(self._safe_literal_eval)

        self.X, self.y = [], []
        neighbor_names = sorted(df['NEIGHBORS'].unique())
        self.neighbor2idx = {n: i for i, n in enumerate(neighbor_names)}

        df['Target'] = pd.to_numeric(df['Target'], errors='coerce')
        df = df.dropna(subset=['Target'])

        for _, row in df.iterrows():
            target = float(row['Target'])
            for n in neighbor_names:
                x = extract_features(row, self.neighbor2idx[n])
                self.X.append(x)
                self.y.append(target)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32).reshape(-1, 1)

        self.x_scaler = StandardScaler()
        self.X = self.x_scaler.fit_transform(self.X)
        self.y_scaler = StandardScaler()
        self.y = self.y_scaler.fit_transform(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor(y, dtype=torch.float32)
    
dataset = NeighborTargetDataset(df)
K = len(dataset.neighbor2idx)
input_dim = dataset.X.shape[1]

class RegNet(nn.Module):
    def __init__(self, in_dim, hid_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1))

    def forward(self, x):
        out = self.net(x)
        if torch.isnan(out).any() or (out.abs() > 1e6).any():
            print("Forward NaN/inf!", out.max().item(), out.min().item())
        return out

model = RegNet(input_dim)
maml  = l2l.algorithms.MAML(model, lr=0.1, first_order=False)

def train_maml(dataset,
               epochs=300,
               meta_lr=0.006,
               fast_lr=0.009,
               adapt_steps=3,
               meta_batch_size=64,
               val_split=0.2,
               seed=42):
    total_len  = len(dataset)
    val_len    = int(total_len * val_split)
    train_len  = total_len - val_len
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_set, batch_size=meta_batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=meta_batch_size, shuffle=False)

    model = RegNet(dataset.X.shape[1])
    maml  = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    optimizer = optim.Adam(maml.parameters(), lr=meta_lr)
    criterion = nn.MSELoss()

    train_rmse_hist, val_rmse_hist = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        meta_loss = 0.0
        for x, y in train_loader:
            learner = maml.clone()
            for _ in range(adapt_steps):
                support_pred = learner(x)
                support_loss = criterion(support_pred, y)
                learner.adapt(support_loss)
            query_pred = learner(x)
            meta_loss += criterion(query_pred, y)

        optimizer.zero_grad()
        meta_loss.backward()
        optimizer.step()

        avg_mse = meta_loss.item() / len(train_loader)
        tr_rmse = np.sqrt(avg_mse * dataset.y_scaler.var_[0])
        train_rmse_hist.append(tr_rmse)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                val_loss += criterion(pred, y).item()

        val_rmse = np.sqrt((val_loss / len(val_loader)) * dataset.y_scaler.var_[0])
        val_rmse_hist.append(val_rmse)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}  |  Train RMSE = {tr_rmse:.2f}  |  Val RMSE = {val_rmse:.2f}")

    return model, dataset.x_scaler, dataset.y_scaler, train_rmse_hist, val_rmse_hist

def predict_best_neighbors(new_instance_dict, model, x_scaler, y_scaler, dataset):
    preds = []
    K = len(dataset.neighbor2idx)
    model.eval()
    with torch.no_grad():
        for n_idx in range(K):
            x = extract_features(new_instance_dict, n_idx)
            x = x_scaler.transform([x])
            x = torch.tensor(x, dtype=torch.float32)
            pred_norm = model(x).item()
            pred_real = y_scaler.inverse_transform([[pred_norm]])[0, 0]
            preds.append(pred_real)

    best_n_idx = int(np.argmin(preds))
    best_name = [n for n, i in dataset.neighbor2idx.items() if i == best_n_idx][0]
    return best_name, preds[best_n_idx]

def clean_saved_model():
    MODEL_PATH = Path("maml_neighbor_model.joblib")
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
        print(f">>>  Old models have been cleared:{MODEL_PATH}")
    else:
        print(">>> No local models to clean up.")

if __name__ == "__main__":
    # clean_saved_model()
    dataset = NeighborTargetDataset(df)
    K = len(dataset.neighbor2idx)

    if MODEL_PATH.exists():
        print(">>> Local model detected; loading directly...")
        model, x_scaler, y_scaler, neighbor2idx, train_rmse_hist, val_rmse_hist = joblib.load(MODEL_PATH)
        dataset.neighbor2idx = neighbor2idx
    else:
        print(">>> No local model detected. Starting training...")
        model, x_scaler, y_scaler, train_rmse_hist, val_rmse_hist = train_maml(dataset)
        to_save = (model, x_scaler, y_scaler, dataset.neighbor2idx, train_rmse_hist, val_rmse_hist)
        joblib.dump(to_save, MODEL_PATH)
        print(f">>> The model has been saved to {MODEL_PATH.resolve()}")

    from auto_instance import new_instance

    name, min_target = predict_best_neighbors(
        new_instance, model, x_scaler, y_scaler, dataset)
    print("Predicting the Best Neighbors:", name, "  Predicted Cost:", min_target)
