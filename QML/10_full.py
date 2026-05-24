import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_covtype, load_digits, make_circles
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer

def save_run_to_csv(filename, history_list):
    """Appends data cleanly. Close the CSV tab in VS Code before running!"""
    if not history_list:
        return
    is_empty = True
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if first_line.strip():
                is_empty = False
    with open(filename, 'a', encoding='utf-8') as f:
        if is_empty:
            headers = ['Model', 'Dataset', 'Qubits', 'Layers', 'Epoch', 'Total Epochs', 
                       'Cost', 'Best Cost', 'Accuracy', 'Best Accuracy', 'Time', 'Total Time']
            f.write(','.join(headers) + '\n')
        for row in history_list:
            row_values = [
                str(row["Model"]),
                str(row["Dataset"]),
                str(row["Qubits"]),
                str(row["Layers"]),
                str(row["Epoch"]),
                str(row["Total Epochs"]),
                f"{row['Cost']:.4f}",
                f"{row['Best Cost']:.4f}",
                f"{row['Accuracy']:.4f}",
                f"{row['Best Accuracy']:.4f}",
                f"{row['Time']:.2f}",
                f"{row['Total Time']:.2f}"
            ]
            f.write(','.join(row_values) + '\n')

        # Clean barrier: Places the line in column 1, adds 8 commas to satisfy the 9-column grid
        f.write('=========' * 9 + '\n')
        f.flush()

# ==========================================
# 1. DATASET LOADER FACTORY
# ==========================================
def load_and_prep_dataset(name):
    print(f"\n--- Loading and Preprocessing: {name} ---")

    if name == "circles":
        X, y = make_circles(n_samples=600, noise=0.15, factor=0.5, random_state=42)
        n_qubits = 3
        X_padded = np.zeros((X.shape[0], 2 ** n_qubits))
        X_padded[:, :X.shape[1]] = X
        X_processed = X_padded
        
    elif name == "digits":
        n_qubits = 6
        data = load_digits()
        df = pd.DataFrame(data.data)
        df['target'] = data.target
        df = df[(df['target'] == 3) | (df['target'] == 8)]
        df['target'] = df['target'].map({3: 0, 8: 1})
        X = df.drop(columns=['target']).values
        y = df['target'].values
        X_processed = StandardScaler().fit_transform(X)
        
    elif name == "covtype":
        n_qubits = 5
        data = fetch_covtype()
        X, y = data.data, data.target
        mask = (y == 1) | (y == 2)
        X, y = X[mask][:600], y[mask][:600] - 1
        
        X_scaled = StandardScaler().fit_transform(X)
        X_processed = PCA(n_components=2**n_qubits).fit_transform(X_scaled)
    
    X_final = Normalizer(norm='l2').fit_transform(X_processed)
    return train_test_split(X_final, y, test_size=0.2, random_state=42), X_final.shape[1], n_qubits

# ==========================================
# MODEL DEFINITIONS
# ==========================================
class ClassicalBaseline(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# --- THE RAW QUANTUM ENGINES ---
class StrongQuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers=4):
        super().__init__()
        dev = qml.device("default.qubit", wires=n_qubits)
        @qml.qnode(dev, interface="torch")
        def qnode(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        self.vqc_layer = qml.qnn.TorchLayer(qnode, {"weights": (n_layers, n_qubits, 3)})
    def forward(self, x): return self.vqc_layer(x)

class BasicQuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers=4):
        super().__init__()
        dev = qml.device("default.qubit", wires=n_qubits)
        @qml.qnode(dev, interface="torch")
        def qnode(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True)
            qml.BasicEntanglerLayers(weights=weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        self.vqc_layer = qml.qnn.TorchLayer(qnode, {"weights": (n_layers, n_qubits)})
    def forward(self, x): return self.vqc_layer(x)

# --- THE PURE VQC WRAPPERS (Averages outputs for binary classification) ---
class StrongQuantumVQC(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.quantum = StrongQuantumLayer(n_qubits)
    def forward(self, x):
        return torch.sigmoid(self.quantum(x).mean(dim=1, keepdim=True))

class BasicQuantumVQC(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.quantum = BasicQuantumLayer(n_qubits)
    def forward(self, x):
        return torch.sigmoid(self.quantum(x).mean(dim=1, keepdim=True))
    
# --- THE HYBRID QNN (Uses clayer_out instead of averaging) ---
class HybridQNN(nn.Module):
    def __init__(self, input_dim, n_qubits, n_layers=4):
        super().__init__()
        self.clayer_in = nn.Linear(input_dim, 2 ** n_qubits)
        self.quantum = StrongQuantumLayer(n_qubits, n_layers)
        self.clayer_out = nn.Linear(n_qubits, 1)

    def forward(self, x):
        x = torch.tanh(self.clayer_in(x))
        x = F.normalize(x, p=2, dim=1)
        x = self.quantum(x) # Outputs shape (Batch, n_qubits)
        x = self.clayer_out(x) # Maps n_qubits down to 1 classical output
        return torch.sigmoid(x)

# ==========================================
# EXECUTION 
# ==========================================
def train_and_evaluate(model, model_name, dataset_name, data_splits, n_qubits):
    if hasattr(model, "quantum") and hasattr(model.quantum, "vqc_layer"):
        n_layers = model.quantum.vqc_layer.qnode_weights["weights"].shape[0]
    elif hasattr(model, "vqc_layer"):
        n_layers = model.vqc_layer.qnode_weights["weights"].shape[0]
    elif model_name == "Classical_DNN" and hasattr(model, "net"):
        n_layers = sum(1 for layer in model.net if isinstance(layer, nn.Linear))
    else:
        n_layers = "N/A"
    
    X_train, X_test, y_train, y_test = data_splits
    
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    Y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    Y_te = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    criterion = nn.BCELoss()
    # Quantum circuits often need a slightly smaller learning rate than purely classical ones
    optimizer = optim.Adam(model.parameters(), lr=0.02) 
    
    history = []
    best_acc = 0.0
    best_cost = 1000.0
    n_epochs = 200
    total_time = 0.0
    
    print(f"  -> Training {model_name}...")
    for epoch in range(n_epochs): # 30 epochs for benchmarking speed
        start_time = time.perf_counter()
        
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr), Y_tr)
        loss.backward()
        optimizer.step()
        
        # Eval Test Acc
        model.eval()
        with torch.no_grad():
            preds = model(X_te)
            acc = ((preds >= 0.5).float() == Y_te).float().mean().item()
            if acc > best_acc: 
                best_acc = acc
            if loss.item() < best_cost:
                best_cost = loss.item()
            
        end_time = time.perf_counter()
        epoch_time = end_time - start_time
        total_time += epoch_time
        
        history.append({
            "Model": model_name,
            "Dataset": dataset_name,
            "Qubits": n_qubits,
            "Layers": n_layers,
            "Epoch": epoch + 1,
            "Total Epochs": n_epochs,
            "Cost": float(loss.item()),
            "Best Cost": float(best_cost),
            "Accuracy": float(acc),
            "Best Accuracy": float(best_acc),
            "Time": (end_time - start_time),
            "Total Time": total_time})
            
        print(f"Epoch {epoch+1:3d} | Cost: {loss.item():.4f} | Test Acc: {acc*100:.1f}% | Time: {end_time - start_time:.2f}s")
    os.makedirs("Outputs", exist_ok=True)
    filename = f"QML/Outputs/03_{model_name}_{dataset_name}_History.csv"
    save_run_to_csv(filename, history)
    print(f"     Finished! Peak Accuracy: {best_acc*100:.2f}%")

# ==========================================
# 4. THE MASTER LOOP
# ==========================================
if __name__ == "__main__":
    datasets = ["circles", "digits", "covtype"]
    
    for target_dataset in datasets:
        splits, input_dim, n_qubits = load_and_prep_dataset(target_dataset)
        
        # Instantiate the 3 competitors
        models = {
            "Classical_DNN": ClassicalBaseline(input_dim),
            "Strong_VQC": StrongQuantumVQC(n_qubits),
            "Basic_VQC":  BasicQuantumVQC(n_qubits),
            "Hybrid_QNN": HybridQNN(input_dim, n_qubits)
        }
        
        for name, model in models.items():
            train_and_evaluate(model, name, target_dataset, splits, n_qubits)