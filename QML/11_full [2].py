import csv
import gzip
import os
import struct
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

RANDOM_SEED = int(os.getenv("QML_RANDOM_SEED", "42"))
TEST_SIZE = float(os.getenv("QML_TEST_SIZE", "0.2"))
N_EPOCHS = int(os.getenv("QML_EPOCHS", "200"))
AVERAGE_RUN_COUNT = int(os.getenv("QML_AVERAGE_RUN_COUNT", "10"))
BATCH_SIZE = int(os.getenv("QML_BATCH_SIZE", "32"))
LEARNING_RATE = float(os.getenv("QML_LEARNING_RATE", "0.02"))
OUTPUT_DIR = os.getenv("QML_OUTPUT_DIR", "QML/Outputs")
DATASETS_TO_RUN = tuple(
    item.strip() for item in os.getenv("QML_DATASETS", "mnist,titanic,cytotoxicity").split(",") if item.strip()
)
MNIST_SAMPLE_SIZE = int(os.getenv("QML_MNIST_SAMPLE_SIZE", "2400"))
MNIST_PCA_COMPONENTS = int(os.getenv("QML_MNIST_PCA_COMPONENTS", "32"))
RUN_GRAPHS = os.getenv("QML_RUN_GRAPHS", "1") != "0"
REFRESH_GRAPHS_AFTER_MODEL = os.getenv("QML_REFRESH_GRAPHS_AFTER_MODEL", "1") != "0"
SHOW_PLOTS = os.getenv("QML_SHOW_PLOTS", "0") == "1"

if not 0.0 < TEST_SIZE < 1.0:
    raise ValueError("QML_TEST_SIZE must be between 0 and 1.")
if N_EPOCHS < 1:
    raise ValueError("QML_EPOCHS must be at least 1.")
if AVERAGE_RUN_COUNT < 1:
    raise ValueError("QML_AVERAGE_RUN_COUNT must be at least 1.")
if BATCH_SIZE < 1:
    raise ValueError("QML_BATCH_SIZE must be at least 1.")
if MNIST_SAMPLE_SIZE < 100:
    raise ValueError("QML_MNIST_SAMPLE_SIZE must be at least 100.")
if MNIST_PCA_COMPONENTS not in (8, 16, 32):
    raise ValueError("QML_MNIST_PCA_COMPONENTS must be 8, 16, or 32 for amplitude embedding.")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


@dataclass(frozen=True)
class DatasetBundle:
    splits: tuple
    input_dim: int
    n_qubits: int
    num_classes: int
    task_type: str
    label: str

def save_run_to_csv(filename, history_list):
    """Appends data cleanly. Close the CSV tab in VS Code before running!"""
    if not history_list:
        return
    output_parent = os.path.dirname(filename)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    is_empty = True
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if first_line.strip():
                is_empty = False
    with open(filename, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if is_empty:
            headers = ['Model', 'Dataset', 'Qubits', 'Layers', 'Epoch', 'Total Epochs', 
                       'Cost', 'Best Cost', 'Accuracy', 'Best Accuracy', 'Time', 'Total Time']
            writer.writerow(headers)
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
            writer.writerow(row_values)

        writer.writerow(['=========' * 9])
        f.flush()


def _model_history_dir(output_dir, model_name):
    return os.path.join(output_dir, f"03_{model_name}_History {{11}}")


def _history_path(output_dir, model_name, dataset_label):
    return os.path.join(
        _model_history_dir(output_dir, model_name),
        f"03_{model_name}_{dataset_label}_History.csv"
    )


def _graph_dir(output_dir):
    return os.path.join(output_dir, "04_Graphs {11}")

# ==========================================
# PREPROCESSING
# ==========================================

def _stratified_split(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

def _normalize_train_test(X_train, X_test):
    normalizer = Normalizer(norm='l2')
    return normalizer.fit_transform(X_train), normalizer.transform(X_test)

def _pad_to_amplitude_dim(X, n_qubits):
    target_dim = 2 ** n_qubits
    if X.shape[1] > target_dim:
        raise ValueError(f"Input dimension {X.shape[1]} exceeds amplitude dimension {target_dim}.")
    X_padded = np.zeros((X.shape[0], target_dim))
    X_padded[:, :X.shape[1]] = X
    return X_padded

def _read_idx_images(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic, n_images, n_rows, n_cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file.")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n_images, n_rows * n_cols).astype(np.float32)

def _read_idx_labels(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic, n_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file.")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    if labels.shape[0] != n_labels:
        raise ValueError(f"{path} label count mismatch.")
    return labels.astype(np.int64)

def _load_local_mnist():
    root = os.path.join(os.path.dirname(__file__), "data", "MNIST", "raw")
    train_images = _read_idx_images(os.path.join(root, "train-images-idx3-ubyte"))
    train_labels = _read_idx_labels(os.path.join(root, "train-labels-idx1-ubyte"))
    test_images = _read_idx_images(os.path.join(root, "t10k-images-idx3-ubyte"))
    test_labels = _read_idx_labels(os.path.join(root, "t10k-labels-idx1-ubyte"))
    X = np.vstack([train_images, test_images])
    y = np.concatenate([train_labels, test_labels])
    if MNIST_SAMPLE_SIZE < len(y):
        rng = np.random.default_rng(RANDOM_SEED)
        keep_indices = []
        per_class = max(1, MNIST_SAMPLE_SIZE // 10)
        for digit in range(10):
            digit_indices = np.flatnonzero(y == digit)
            take = min(per_class, len(digit_indices))
            keep_indices.extend(rng.choice(digit_indices, size=take, replace=False))
        if len(keep_indices) < MNIST_SAMPLE_SIZE:
            remaining = np.setdiff1d(np.arange(len(y)), np.array(keep_indices), assume_unique=False)
            extra = rng.choice(remaining, size=MNIST_SAMPLE_SIZE - len(keep_indices), replace=False)
            keep_indices.extend(extra)
        keep_indices = np.array(keep_indices)
        rng.shuffle(keep_indices)
        X, y = X[keep_indices], y[keep_indices]
    return X, y

def _prep_scaled_pca_dataset(X, y, n_qubits, n_components, label, task_type, num_classes):
    X_train, X_test, y_train, y_test = _stratified_split(X, y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if n_components is not None:
        pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    X_train, X_test = _normalize_train_test(X_train, X_test)
    X_train = _pad_to_amplitude_dim(X_train, n_qubits)
    X_test = _pad_to_amplitude_dim(X_test, n_qubits)
    return DatasetBundle(
        splits=(X_train, X_test, y_train.astype(np.int64), y_test.astype(np.int64)),
        input_dim=X_train.shape[1],
        n_qubits=n_qubits,
        num_classes=num_classes,
        task_type=task_type,
        label=label,
    )

def _load_titanic_features():
    path = os.path.join(os.path.dirname(__file__), "data", "train.csv")
    df = pd.read_csv(path)
    y = df["Survived"].to_numpy(dtype=np.int64)
    features = pd.DataFrame({
        "Pclass": df["Pclass"],
        "Sex": df["Sex"].map({"male": 0.0, "female": 1.0}),
        "Age": df["Age"],
        "SibSp": df["SibSp"],
        "Parch": df["Parch"],
        "Fare": df["Fare"],
        "FamilySize": df["SibSp"] + df["Parch"] + 1,
        "IsAlone": ((df["SibSp"] + df["Parch"]) == 0).astype(float),
        "HasCabin": df["Cabin"].notna().astype(float),
        "Embarked": df["Embarked"].fillna("Missing"),
        "Title": df["Name"].str.extract(r",\s*([^.]*)\.", expand=False).fillna("Unknown"),
    })
    for column in ["Age", "Fare"]:
        features.loc[:, column] = features[column].fillna(features[column].median())
    features = pd.get_dummies(features, columns=["Embarked", "Title"], drop_first=False)
    return features.to_numpy(dtype=np.float32), y

def _load_cytotoxicity_features():
    root = os.path.join(os.path.dirname(__file__), "data_tutorial", "pqk")
    frames = []
    for filename in ("train_data.csv", "test_data.csv", "noisy_data.csv"):
        path = os.path.join(root, filename)
        raw = pd.read_csv(path, encoding="utf-8-sig")
        if raw.shape[1] == 1:
            header = raw.columns[0].split(",")
            parsed = raw.iloc[:, 0].astype(str).str.split(",", expand=True)
            parsed.columns = header
        else:
            parsed = raw.copy()
        frames.append(parsed)
    df = pd.concat(frames, ignore_index=True)
    for column in df.columns:
        df.loc[:, column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna().drop_duplicates()
    target = df["Nalm 6 Cytotoxicity"].to_numpy(dtype=np.float32)
    threshold = float(np.median(target))
    y = (target >= threshold).astype(np.int64)
    X = df.drop(columns=["Nalm 6 Cytotoxicity"]).to_numpy(dtype=np.float32)
    return X, y

def load_and_prep_dataset(name):
    print(f"\n--- Loading and Preprocessing: {name} ---")

    if name == "mnist":
        X, y = _load_local_mnist()
        n_qubits = int(np.log2(MNIST_PCA_COMPONENTS))
        return _prep_scaled_pca_dataset(
            X=X,
            y=y,
            n_qubits=n_qubits,
            n_components=MNIST_PCA_COMPONENTS,
            label=f"mnist_10class_pca{MNIST_PCA_COMPONENTS}",
            task_type="multiclass",
            num_classes=10,
        )

    elif name == "titanic":
        X, y = _load_titanic_features()
        return _prep_scaled_pca_dataset(
            X=X,
            y=y,
            n_qubits=3,
            n_components=8,
            label="titanic_survival",
            task_type="binary",
            num_classes=2,
        )

    elif name == "cytotoxicity":
        X, y = _load_cytotoxicity_features()
        return _prep_scaled_pca_dataset(
            X=X,
            y=y,
            n_qubits=3,
            n_components=None,
            label="cytotoxicity_median",
            task_type="binary",
            num_classes=2,
        )

    raise ValueError(f"Unknown dataset: {name}")

# ==========================================
# MODEL DEFINITIONS
# ==========================================

class ClassicalBaseline(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
    def forward(self, x): 
        return self.net(x)

# --- QUANTUM ENGINES ---
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
    def forward(self, x): 
        return self.vqc_layer(x)

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
    def forward(self, x): 
        return self.vqc_layer(x)

# --- THE PURE VQC WRAPPERS ---
class StrongQuantumVQC(nn.Module):
    def __init__(self, n_qubits, output_dim):
        super().__init__()
        self.quantum = StrongQuantumLayer(n_qubits)
        self.readout = nn.Linear(n_qubits, output_dim, bias=False)
    def forward(self, x):
        return self.readout(self.quantum(x))

class BasicQuantumVQC(nn.Module):
    def __init__(self, n_qubits, output_dim):
        super().__init__()
        self.quantum = BasicQuantumLayer(n_qubits)
        self.readout = nn.Linear(n_qubits, output_dim, bias=False)
    def forward(self, x):
        return self.readout(self.quantum(x))
    
# --- HYBRID QNN  ---
class HybridQNN(nn.Module):
    def __init__(self, input_dim, n_qubits, output_dim, n_layers=4):
        super().__init__()
        # Bias-free bottleneck keeps the hybrid path from using classical offsets as a crutch.
        hidden_dim = min(input_dim, n_qubits * 2)
        self.clayer_in = nn.Linear(input_dim, hidden_dim, bias=False)
        self.quantum_prepare = nn.Linear(hidden_dim, 2 ** n_qubits, bias=False)
        self.quantum = StrongQuantumLayer(n_qubits, n_layers)
        self.clayer_out = nn.Linear(n_qubits, output_dim, bias=False)

    def forward(self, x):
        x = torch.tanh(self.clayer_in(x))
        x = torch.tanh(self.quantum_prepare(x))
        x = F.normalize(x, p=2, dim=1)
        x = self.quantum(x) 
        x = self.clayer_out(x) 
        return x

# ==========================================
# EXECUTION 
# ==========================================

def _output_dim_for_task(num_classes, task_type):
    return 1 if task_type == "binary" else num_classes

def _targets_to_tensor(y, task_type):
    if task_type == "binary":
        return torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return torch.tensor(y, dtype=torch.long)

def _criterion_for_task(task_type):
    if task_type == "binary":
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()

def _accuracy_from_logits(logits, targets, task_type):
    if task_type == "binary":
        preds = (torch.sigmoid(logits) >= 0.5).float()
        return (preds == targets).float().mean().item()
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()

def train_and_evaluate(model, model_name, dataset_bundle, output_dir=OUTPUT_DIR):
    if hasattr(model, "quantum") and hasattr(model.quantum, "vqc_layer"):
        n_layers = model.quantum.vqc_layer.qnode_weights["weights"].shape[0]
    elif hasattr(model, "vqc_layer"):
        n_layers = model.vqc_layer.qnode_weights["weights"].shape[0]
    elif model_name == "Classical_DNN" and hasattr(model, "net"):
        n_layers = sum(1 for layer in model.net if isinstance(layer, nn.Linear))
    else:
        n_layers = "N/A"
    
    X_train, X_test, y_train, y_test = dataset_bundle.splits
    task_type = dataset_bundle.task_type
    
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    Y_tr = _targets_to_tensor(y_train, task_type)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    Y_te = _targets_to_tensor(y_test, task_type)
    
    criterion = _criterion_for_task(task_type)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    
    history = []
    best_acc = 0.0
    best_cost = float("inf")
    total_time = 0.0
    
    dataset = TensorDataset(X_tr, Y_tr)
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)

    print(f"  -> Training {model_name} ({task_type}, {dataset_bundle.num_classes} classes)...")
    for epoch in range(N_EPOCHS):
        start_time = time.perf_counter()

        model.train()
        train_loss_sum = 0.0
        for X_batch, Y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), Y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * X_batch.shape[0]
        train_loss = train_loss_sum / len(dataset)
        
        model.eval()
        with torch.no_grad():
            test_logits = model(X_te)
            acc = _accuracy_from_logits(test_logits, Y_te, task_type)
            test_loss = criterion(test_logits, Y_te).item()
            if acc > best_acc: 
                best_acc = acc
            if train_loss < best_cost:
                best_cost = train_loss
            
        end_time = time.perf_counter()
        epoch_time = end_time - start_time
        total_time += epoch_time
        
        history.append({
            "Model": model_name,
            "Dataset": dataset_bundle.label,
            "Qubits": dataset_bundle.n_qubits,
            "Layers": n_layers,
            "Epoch": epoch + 1,
            "Total Epochs": N_EPOCHS,
            "Cost": float(train_loss),
            "Best Cost": float(best_cost),
            "Accuracy": float(acc),
            "Best Accuracy": float(best_acc),
            "Time": (end_time - start_time),
            "Total Time": total_time})
            
        print(
            f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {acc*100:.1f}% | "
            f"Time: {end_time - start_time:.2f}s"
        )
    filename = _history_path(output_dir, model_name, dataset_bundle.label)
    save_run_to_csv(filename, history)
    print(f"     Finished! Peak Accuracy: {best_acc*100:.2f}%")

#=========================================
# GRAPHS
#=========================================

def get_clean_latest_run(file_path):
    runs = get_clean_complete_runs(file_path, expected_epochs=None)
    if not runs:
        return None
    return runs[-1]

def _csv_rows_to_history_df(rows, header):
    cleaned_rows = []
    for row_values in rows:
        if not row_values or any("===" in value for value in row_values):
            continue
        if len(row_values) == len(header):
            cleaned_rows.append(row_values)
    if not cleaned_rows:
        return None

    df = pd.DataFrame(cleaned_rows, columns=header)
    numeric_columns = ["Epoch", "Total Epochs", "Cost", "Best Cost", "Accuracy", "Best Accuracy", "Time", "Total Time"]
    for column in numeric_columns:
        if column in df.columns:
            df.loc[:, column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["Epoch", "Accuracy"]).sort_values("Epoch")
    if df.empty:
        return None
    return df

def get_clean_complete_runs(file_path, expected_epochs=N_EPOCHS):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    header = rows[0]
    split_indices = [
        i for i, row in enumerate(rows)
        if row and any("===" in value for value in row)
    ]

    segments = []
    start_index = 1
    for split_index in split_indices:
        segments.append(rows[start_index:split_index])
        start_index = split_index + 1
    if start_index < len(rows):
        segments.append(rows[start_index:])
    if not segments:
        segments = [rows[1:]]

    runs = []
    for segment in segments:
        df = _csv_rows_to_history_df(segment, header)
        if df is None:
            continue
        if expected_epochs is not None:
            df = df[df["Epoch"].between(1, expected_epochs)].copy()
            df = df.drop_duplicates(subset=["Epoch"], keep="last").sort_values("Epoch")
            complete_epochs = set(range(1, expected_epochs + 1))
            if set(df["Epoch"].astype(int)) != complete_epochs:
                continue
            df = df.iloc[:expected_epochs].copy()
        runs.append(df)
    return runs

def get_average_complete_run(file_path, expected_epochs=N_EPOCHS):
    runs = get_clean_complete_runs(file_path, expected_epochs=expected_epochs)
    if not runs:
        return None
    runs = runs[-AVERAGE_RUN_COUNT:]

    summaries = []
    for instance, run in enumerate(runs, start=1):
        accuracy = run["Accuracy"].astype(float)
        if accuracy.max() <= 1.0:
            accuracy = accuracy * 100
        summaries.append({
            "Instance": instance,
            "Accuracy": float(accuracy.mean()),
            "Accuracy Std": float(accuracy.std(ddof=0)),
            "Run Count": len(runs),
        })
    return pd.DataFrame(summaries)

def _load_graph_data(paths, labels, dataset_name, use_average=False):
    loaded_datasets = {}
    for label, path in zip(labels, paths):
        if use_average:
            df = get_average_complete_run(path)
            missing_message = f"No complete {N_EPOCHS}-epoch runs found"
        else:
            df = get_clean_latest_run(path)
            missing_message = "Clean latest log sequence not found"
        if df is not None and not df.empty:
            loaded_datasets[label] = df
        else:
            print(f"Skipping model {label} for {dataset_name}: {missing_message} at {path}")
    if len(loaded_datasets) < 4:
        print(f"Aborting plot generation for {dataset_name}: One or more model data components are missing or unreadable.\n")
        return None
    return loaded_datasets

def save_average_accuracy_csv(dataset_name, labels, acc_data, run_counts, output_dir=OUTPUT_DIR):
    graph_dir = _graph_dir(output_dir)
    os.makedirs(graph_dir, exist_ok=True)
    clean_name = dataset_name.lower().replace(" ", "_")
    max_instances = max(len(values) for values in acc_data.values())
    rows = {"Instance": np.arange(1, max_instances + 1)}
    for label in labels:
        values = np.asarray(acc_data[label], dtype=float)
        rows[f"{label} Average Accuracy"] = np.round(
            np.pad(values, (0, max_instances - len(values)), constant_values=np.nan),
            2,
        )
        rows[f"{label} Complete Runs"] = run_counts[label]
    output_path = os.path.join(graph_dir, f"04_Average_Accuracy_{clean_name}.csv")
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path

def generate_dataset_diff_graph(dataset_name, csv_path_a, csv_path_b, csv_path_c, csv_path_d, 
                                label_a, label_b, label_c, label_d, output_dir=OUTPUT_DIR,
                                use_average=False):
    paths = [csv_path_a, csv_path_b, csv_path_c, csv_path_d]
    labels = [label_a, label_b, label_c, label_d]
    loaded_datasets = _load_graph_data(paths, labels, dataset_name, use_average=use_average)
    if loaded_datasets is None:
        return
    min_epochs = min(len(df) for df in loaded_datasets.values())
    x_values = np.arange(1, min_epochs + 1)
    acc_data = {}
    run_counts = {}
    for label, df in loaded_datasets.items():
        acc = df["Accuracy"].iloc[:min_epochs].values
        if acc.max() <= 1.0: 
            acc = acc * 100
        acc_data[label] = acc
        run_counts[label] = int(df["Run Count"].iloc[:min_epochs].min()) if "Run Count" in df.columns else 1

    design_palette = {
        label_a: {"color": "darkorange", "ls": "-.", "marker": "s"}, 
        label_b: {"color": "purple",     "ls": ":",  "marker": "x"}, 
        label_c: {"color": "crimson",    "marker": "o", "ls": "-"},  
        label_d: {"color": "royalblue",  "marker": "^", "ls": "--"}  
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2, 1.1]})
    graph_title = "Average Accuracy Benchmark" if use_average else "Empirical Benchmark Matrix"
    fig.suptitle(f"{graph_title}: {dataset_name}", fontsize=16, fontweight="bold", y=0.98)

    for lbl in labels:
        ax1.plot(x_values, acc_data[lbl], label=lbl, 
                 color=design_palette[lbl]["color"], 
                 linestyle=design_palette[lbl]["ls"], 
                 linewidth=2,
                 marker=design_palette[lbl]["marker"], 
                 markevery=max(1, min_epochs//10), 
                 markersize=6)
        
    if use_average:
        run_summary = ", ".join(f"{lbl}: n={run_counts[lbl]}" for lbl in labels)
        ax1.set_title(f"Per-Instance Mean Accuracy Across {N_EPOCHS} Epochs\n{run_summary}", fontsize=10, style="italic")
    else:
        ax1.set_title("Generalization Accuracy Trajectory Over Epochs", fontsize=11, style="italic")
    ax1.set_xlabel("Training Instance" if use_average else "Epoch", fontsize=11)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    if use_average:
        ax1.set_xticks(x_values)
    elif min_epochs > 20:
        ticks = [1] + list(range(10, min_epochs + 1, 10))
        ax1.set_xticks(ticks)
    else:
        ax1.set_xticks(x_values)
        
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="lightgray", fontsize=10)

    x_bars = np.arange(len(labels))
    bar_width = 0.35

    if use_average:
        mean_accs = [float(np.mean(acc_data[lbl])) for lbl in labels]
        std_accs = [float(np.std(acc_data[lbl], ddof=0)) for lbl in labels]
        bar1 = ax2.bar(x_bars, mean_accs, bar_width, yerr=std_accs, label="Mean +/- SD", color="forestgreen", alpha=0.85, edgecolor="black", linewidth=0.7, capsize=4)
        comparison_values = mean_accs + list(np.concatenate(list(acc_data.values())))
        ax2.set_title("Instance-Level Stability", fontsize=11, style="italic")
    else:
        mean_accs = [float(np.mean(acc_data[lbl])) for lbl in labels]
        std_accs = [float(np.std(acc_data[lbl], ddof=0)) for lbl in labels]
        bar1 = ax2.bar(x_bars, mean_accs, bar_width, yerr=std_accs, label="Mean +/- SD", color="forestgreen", alpha=0.85, edgecolor="black", linewidth=0.7, capsize=4)
        comparison_values = mean_accs + list(np.concatenate(list(acc_data.values())))
        ax2.set_title("Training-Integrated Stability", fontsize=11, style="italic")

    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_xticks(x_bars)
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.4, axis="y")
    ax2.legend(loc="lower left", fontsize=9)
    
    for bar in bar1:
        height = bar.get_height()
        ax2.annotate(f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    all_values = comparison_values
    ymin = max(0, min(all_values) - 5)
    ymax = min(100, max(all_values) + 10)
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)
    plt.tight_layout()
    
    graph_dir = _graph_dir(output_dir)
    os.makedirs(graph_dir, exist_ok=True)
    clean_name = dataset_name.lower().replace(" ", "_")
    filename_prefix = "04_Average_Graph" if use_average else "04_Graph"
    save_filename = os.path.join(graph_dir, f"{filename_prefix}_{clean_name}.png")
    plt.savefig(save_filename, dpi=300)
    if SHOW_PLOTS:
        print(f"Displaying graph pop-up for {dataset_name}...")
        plt.show() 
    plt.close()
    if use_average:
        average_csv = save_average_accuracy_csv(dataset_name, labels, acc_data, run_counts, output_dir=output_dir)
        print(f"Success! Exported averaged benchmark graph for {dataset_name} to {save_filename}")
        print(f"         Exported averaged accuracy table to {average_csv}\n")
    else:
        print(f"Success! Exported comprehensive benchmark analysis for {dataset_name} to {save_filename}\n")

def build_models(input_dim, n_qubits, output_dim):
    return {
        "Classical_DNN": ClassicalBaseline(input_dim, output_dim),
        "Strong_VQC": StrongQuantumVQC(n_qubits, output_dim),
        "Basic_VQC": BasicQuantumVQC(n_qubits, output_dim),
        "Hybrid_QNN": HybridQNN(input_dim, n_qubits, output_dim)
    }

def run_benchmarks(datasets=DATASETS_TO_RUN, output_dir=OUTPUT_DIR):
    completed_labels = {}
    for target_dataset in datasets:
        dataset_bundle = load_and_prep_dataset(target_dataset)
        completed_labels[target_dataset] = dataset_bundle.label
        output_dim = _output_dim_for_task(dataset_bundle.num_classes, dataset_bundle.task_type)
        print(
            f"Prepared {dataset_bundle.label}: train={len(dataset_bundle.splits[0])}, "
            f"test={len(dataset_bundle.splits[1])}, input_dim={dataset_bundle.input_dim}, "
            f"qubits={dataset_bundle.n_qubits}, task={dataset_bundle.task_type}"
        )
        models = build_models(dataset_bundle.input_dim, dataset_bundle.n_qubits, output_dim)
        for name, model in models.items():
            train_and_evaluate(model, name, dataset_bundle, output_dir=output_dir)
            if RUN_GRAPHS and REFRESH_GRAPHS_AFTER_MODEL:
                generate_all_graphs(
                    completed_labels=completed_labels,
                    output_dir=output_dir,
                    only_dataset_names={_display_name_for_dataset(target_dataset)}
                )
    return completed_labels

def _display_name_for_dataset(dataset_key):
    return {
        "mnist": "MNIST 10-Class",
        "titanic": "Titanic Survival",
        "cytotoxicity": "Cytotoxicity Median",
    }.get(dataset_key, dataset_key)

def build_dataset_benchmarks(completed_labels=None, output_dir=OUTPUT_DIR):
    completed_labels = completed_labels or {}
    mnist_label = completed_labels.get("mnist", f"mnist_10class_pca{MNIST_PCA_COMPONENTS}")
    titanic_label = completed_labels.get("titanic", "titanic_survival")
    cytotoxicity_label = completed_labels.get("cytotoxicity", "cytotoxicity_median")
    return {
        "MNIST 10-Class": {
            "a": _history_path(output_dir, "Basic_VQC", mnist_label),
            "b": _history_path(output_dir, "Classical_DNN", mnist_label),
            "c": _history_path(output_dir, "Hybrid_QNN", mnist_label),
            "d": _history_path(output_dir, "Strong_VQC", mnist_label)
        },
        "Titanic Survival": {
            "a": _history_path(output_dir, "Basic_VQC", titanic_label),
            "b": _history_path(output_dir, "Classical_DNN", titanic_label),
            "c": _history_path(output_dir, "Hybrid_QNN", titanic_label),
            "d": _history_path(output_dir, "Strong_VQC", titanic_label)
        },
        "Cytotoxicity Median": {
            "a": _history_path(output_dir, "Basic_VQC", cytotoxicity_label),
            "b": _history_path(output_dir, "Classical_DNN", cytotoxicity_label),
            "c": _history_path(output_dir, "Hybrid_QNN", cytotoxicity_label),
            "d": _history_path(output_dir, "Strong_VQC", cytotoxicity_label)
        }
    }

def generate_all_graphs(completed_labels=None, output_dir=OUTPUT_DIR, only_dataset_names=None):
    print("Processing Dataset Difference Matrices Interactively...")
    for name, paths in build_dataset_benchmarks(completed_labels, output_dir).items():
        if only_dataset_names is not None and name not in only_dataset_names:
            continue
        generate_dataset_diff_graph(
            dataset_name=name,
            csv_path_a=paths["a"],
            csv_path_b=paths["b"],
            csv_path_c=paths["c"],
            csv_path_d=paths["d"],
            label_a="Basic VQC",
            label_b="Classical DNN",
            label_c="Hybrid QNN",
            label_d="Strong VQC",
            output_dir=output_dir
        )
        generate_dataset_diff_graph(
            dataset_name=name,
            csv_path_a=paths["a"],
            csv_path_b=paths["b"],
            csv_path_c=paths["c"],
            csv_path_d=paths["d"],
            label_a="Basic VQC",
            label_b="Classical DNN",
            label_c="Hybrid QNN",
            label_d="Strong VQC",
            output_dir=output_dir,
            use_average=True
        )

def main():
    completed_labels = run_benchmarks()
    if RUN_GRAPHS:
        generate_all_graphs(completed_labels)

if __name__ == "__main__":
    main()
