import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch, torch.nn as nn, learn2learn as l2l
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

MODEL_PATH = Path("maml_GC_multiout_model.joblib")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

df = pd.read_csv(r"test_CM\result_GC.csv")
combos = df[['MAX_ITER', 'MAX_NEW_COLS']].drop_duplicates().sort_values(['MAX_ITER', 'MAX_NEW_COLS'])

def extract_features(row, max_iter, max_new_cols):
    scalar = [row['N'], row['T'], row['seta'], row['c'], max_iter, max_new_cols]
    seq_stats = []
    for col in ['r_j','t_k','p_j','L_k','S_k','CP_k']:
        v = eval(row[col])
        seq_stats += [np.mean(v), np.std(v), np.max(v), np.min(v)]
    return np.array(scalar + seq_stats, dtype=np.float32)

class TargetDataset(Dataset):
    def __init__(self, df, x_scaler=None, y_scaler=None):
        self.X, self.y = [], []
        df = df.copy()
        df['Target'] = pd.to_numeric(df['Target'], errors='coerce')
        df[['HU_cost','HC_cost','HM_cost']] = df[['HU_cost','HC_cost','HM_cost']].apply(
            pd.to_numeric, errors='coerce')
        df = df.dropna(subset=['Target','HU_cost','HC_cost','HM_cost'])

        combos = df[['MAX_ITER', 'MAX_NEW_COLS']].drop_duplicates()
        for _, row in df.iterrows():
            targets = [float(row['Target']),
                       float(row['HU_cost']),
                       float(row['HC_cost']),
                       float(row['HM_cost'])]
            for _, combo in combos.iterrows():
                x = extract_features(row, combo['MAX_ITER'], combo['MAX_NEW_COLS'])
                self.X.append(x)
                self.y.append(targets)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32).reshape(-1, 4)

        if x_scaler is None:
            self.x_scaler = StandardScaler()
            self.X = self.x_scaler.fit_transform(self.X)
        else:
            self.x_scaler = x_scaler
            self.X = self.x_scaler.transform(self.X)

        if y_scaler is None:
            self.y_scaler = StandardScaler()
            self.y = self.y_scaler.fit_transform(self.y)
        else:
            self.y_scaler = y_scaler
            self.y = self.y_scaler.transform(self.y)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

class EarlyStopping:
    def __init__(self, patience=20, delta=0, trace=True):
        self.patience = patience
        self.delta = delta
        self.trace = trace
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_rmse, model):
        if self.best_score is None:
            self.best_score = val_rmse
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_rmse < self.best_score - self.delta:
            self.best_score = val_rmse
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.trace:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}  "
                      f"(best={self.best_score:.4f}, curr={val_rmse:.4f}, "
                      f"delta<{self.delta})")
        if self.counter >= self.patience:
            self.early_stop = True
            if self.trace:
                print("EarlyStopping triggered")
        return self.early_stop

    def load_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

class RegNet(nn.Module):
    def __init__(self, in_dim, hid_dim=256, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
                    nn.Linear(in_dim, hid_dim),
                    nn.GroupNorm(8, hid_dim),
                    nn.ReLU(),
                    nn.Linear(hid_dim, out_dim))
    def forward(self, x): return self.net(x)

def validate_one_epoch(model, val_loader, criterion, y_scaler):
    model.eval()
    val_mse = 0.0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            y_pred = model(x_val)
            val_mse += criterion(y_pred, y_val).item()
    return np.sqrt(val_mse / len(val_loader) * y_scaler.var_[0])

def train_maml(dataset,
               epochs=300,
               meta_lr=0.01,
               fast_lr=0.01,
               adapt_steps=5,
               meta_batch_size=64,
               val_split=0.2,
               seed=42):
    total_len = len(dataset)
    val_len   = int(total_len * val_split)
    train_len = total_len - val_len
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
        real_rmse = np.sqrt(avg_mse * dataset.y_scaler.var_[0])
        train_rmse_hist.append(real_rmse)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                val_loss += criterion(pred, y).item()

        val_rmse = np.sqrt((val_loss / len(val_loader)) * dataset.y_scaler.var_[0])
        val_rmse_hist.append(val_rmse)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}  |  Train RMSE = {real_rmse:.2f}  |  Val RMSE = {val_rmse:.2f}")

    return model, dataset.x_scaler, dataset.y_scaler, train_rmse_hist, val_rmse_hist

def predict_best_combo(new_instance_dict, model, x_scaler, y_scaler):
    best_target = float('inf')
    best_combo = (None, None)
    best_costs = None
    model.eval()
    with torch.no_grad():
        for _, row in combos.iterrows():
            max_iter, max_new_cols = row['MAX_ITER'], row['MAX_NEW_COLS']
            x = extract_features(new_instance_dict, max_iter, max_new_cols)
            x = x_scaler.transform([x])
            pred_norm = model(torch.tensor(x, dtype=torch.float32)).squeeze(0).numpy()
            pred_real = y_scaler.inverse_transform([pred_norm])[0]
            if pred_real[0] < best_target:
                best_target = pred_real[0]
                best_combo = (max_iter, max_new_cols)
                best_costs = pred_real[1:]
    return best_combo, best_target, best_costs

def clean_saved_model(path):
    MODEL_PATH = Path(path)
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
        print(f">>> Old models have been cleared: {MODEL_PATH}")
    else:
        print(">>> No local models to clean up.")

def plot_rmse_curve(train_rmse, val_rmse, save_path=None):
    cm = 1 / 2.54
    plt.figure(figsize=(7.1 * cm, 6 * cm))
    plt.rcParams.update({'font.size': 7})

    epochs = np.arange(1, len(train_rmse) + 1)
    ax = plt.gca()
    ax.set_facecolor('#f5f0ff')
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.plot(epochs, train_rmse, label='Train RMSE', color='tab:blue', linewidth=1)
    plt.plot(epochs, val_rmse, label='Val RMSE', color='tab:orange', linewidth=1)

    best_tr, best_val = np.min(train_rmse), np.min(val_rmse)
    plt.scatter(epochs[np.argmin(train_rmse)], best_tr, s=10, color='tab:blue', zorder=3)
    plt.scatter(epochs[np.argmin(val_rmse)], best_val, s=10, color='tab:orange', zorder=3)

    plt.xlabel('Epoch', fontsize=8, labelpad=3)
    plt.ylabel('RMSE', fontsize=8, labelpad=2)
    plt.title('CGDH-MAML RMSE Convergence Curve', fontsize=8, pad=5)

    legend = plt.legend()
    legend.get_frame().set_facecolor('#f5f0ff')
    legend.get_frame().set_edgecolor('white')

    ax.grid(alpha=0.7, color='white')
    plt.tight_layout(pad=0.1)
    plt.tight_layout(pad=0.1)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f"RMSE 收敛图已保存 → {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # clean_saved_model(MODEL_PATH) 

    if MODEL_PATH.exists():
        print(">>> Local model detected; loading directly...")
        model, x_scaler, y_scaler, train_hist, val_hist = joblib.load(MODEL_PATH)
        
        # plot_rmse_curve(train_hist, val_hist, save_path="rmse_curve.png")

    else:
        print(">>> No model detected. Starting training...")
        
        dataset = TargetDataset(df)
        model, x_scaler, y_scaler, train_hist, val_hist = train_maml(dataset)
        joblib.dump((model, x_scaler, y_scaler, train_hist, val_hist), MODEL_PATH)
        print(">>> The model has been saved.")

    plot_rmse_curve(train_hist, val_hist, save_path="rmse_curve.png")

    from auto_instance import new_instance
    best_combo, min_target, best_costs = predict_best_combo(new_instance, model, x_scaler, y_scaler)

    cost_names = ['HU', 'HC', 'HM']
    min_index = np.argmin(best_costs)
    suggested_strategy = cost_names[min_index]

    print("Best combo:", best_combo)
    print("Predicted Minimum Cost:", min_target)
    print("Predicted Cost_HU/HC/HM:", best_costs)
    print("Suggested strategy:", suggested_strategy)

