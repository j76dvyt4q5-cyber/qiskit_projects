import csv
import os
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
from sklearn.datasets import fetch_covtype, load_digits, make_circles
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

def _resolve_output_dir(output_dir):
    expanded = os.path.expanduser(output_dir)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    normalized = os.path.normpath(expanded)
    if normalized == "QML" or normalized.startswith(f"QML{os.sep}"):
        return os.path.normpath(os.path.join(PROJECT_DIR, normalized))
    return os.path.normpath(os.path.join(SCRIPT_DIR, normalized))

RANDOM_SEED = int(os.getenv("QML_RANDOM_SEED", "42"))
TEST_SIZE = float(os.getenv("QML_TEST_SIZE", "0.2"))
N_EPOCHS = int(os.getenv("QML_EPOCHS", "200"))
AVERAGE_RUN_COUNT = int(os.getenv("QML_AVERAGE_RUN_COUNT", "10"))
BATCH_SIZE = int(os.getenv("QML_BATCH_SIZE", "32"))
LEARNING_RATE = float(os.getenv("QML_LEARNING_RATE", "0.02"))
OUTPUT_DIR = _resolve_output_dir(os.getenv("QML_OUTPUT_DIR", "Outputs"))
DATASETS_TO_RUN = tuple(
    item.strip() for item in os.getenv("QML_DATASETS", "circles,digits,covtype").split(",") if item.strip()
)
DIGIT_CLASSES = tuple(
    int(item.strip()) for item in os.getenv("QML_DIGIT_CLASSES", "0,1,2,3,4,5,6,7,8,9").split(",") if item.strip()
)
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
if len(set(DIGIT_CLASSES)) != len(DIGIT_CLASSES):
    raise ValueError("QML_DIGIT_CLASSES must not contain duplicates.")
if any(digit < 0 or digit > 9 for digit in DIGIT_CLASSES):
    raise ValueError("QML_DIGIT_CLASSES may only contain digits 0 through 9.")

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

        # Clean barrier used by get_clean_latest_run to isolate the newest complete run.
        writer.writerow(['=========' * 9])
        f.flush()


def _model_history_dir(output_dir, model_name):
    output_dir = _resolve_output_dir(output_dir)
    return os.path.join(output_dir, f"03_{model_name}_History {{10}}")


def _history_path(output_dir, model_name, dataset_label):
    return os.path.join(
        _model_history_dir(output_dir, model_name),
        f"03_{model_name}_{dataset_label}_History.csv"
    )


def _graph_dir(output_dir):
    output_dir = _resolve_output_dir(output_dir)
    return os.path.join(output_dir, "04_Graphs {10}")


def _cancer_history_path(output_dir, filename):
    output_dir = _resolve_output_dir(output_dir)
    return os.path.join(output_dir, "02_CancerHistory {All}", filename)

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

def _digit_label(classes):
    if tuple(classes) == tuple(range(10)):
        return "digits_0-9"
    return "digits_" + "-".join(str(item) for item in classes)

def load_and_prep_dataset(name):
    print(f"\n--- Loading and Preprocessing: {name} ---")

    if name == "circles":
        X, y = make_circles(
        n_samples=600,
        noise=0.15,
        factor=0.5,
        random_state=RANDOM_SEED)
        
        n_qubits = 3
        X_train, X_test, y_train, y_test = _stratified_split(X, y)
        X_train, X_test = _normalize_train_test(X_train, X_test)
        X_train = _pad_to_amplitude_dim(X_train, n_qubits)
        X_test = _pad_to_amplitude_dim(X_test, n_qubits)

        return DatasetBundle(
            splits=(X_train, X_test, y_train.astype(np.int64), y_test.astype(np.int64)),
            input_dim=X_train.shape[1],
            n_qubits=n_qubits,
            num_classes=2,
            task_type="binary",
            label="circles",
        )
        
    elif name == "digits":
        if len(DIGIT_CLASSES) < 2:
            raise ValueError("QML_DIGIT_CLASSES must contain at least two digit labels.")
        n_qubits = 6
        data = load_digits()
        df = pd.DataFrame(data.data)
        df['target'] = data.target
        df = df[df['target'].isin(DIGIT_CLASSES)].copy()
        if df.empty:
            raise ValueError(f"No digit samples found for QML_DIGIT_CLASSES={DIGIT_CLASSES}.")
        class_map = {digit: idx for idx, digit in enumerate(DIGIT_CLASSES)}
        df.loc[:, 'target'] = df['target'].map(class_map)
        X = df.drop(columns=['target']).values
        y = df['target'].to_numpy(dtype=np.int64)
        X_train, X_test, y_train, y_test = _stratified_split(X, y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train, X_test = _normalize_train_test(X_train, X_test)
        return DatasetBundle(
            splits=(X_train, X_test, y_train, y_test),
            input_dim=X_train.shape[1],
            n_qubits=n_qubits,
            num_classes=len(DIGIT_CLASSES),
            task_type="binary" if len(DIGIT_CLASSES) == 2 else "multiclass",
            label=_digit_label(DIGIT_CLASSES),
        )
        
    elif name == "covtype":
        n_qubits = 5
        data = fetch_covtype()
        X, y = data.data, data.target
        mask = (y == 1) | (y == 2)
        X, y = X[mask][:600], (y[mask][:600] - 1).astype(np.int64)
        X_train, X_test, y_train, y_test = _stratified_split(X, y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        pca = PCA(n_components=2**n_qubits, random_state=RANDOM_SEED)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        X_train, X_test = _normalize_train_test(X_train, X_test)
        return DatasetBundle(
            splits=(X_train, X_test, y_train, y_test),
            input_dim=X_train.shape[1],
            n_qubits=n_qubits,
            num_classes=2,
            task_type="binary",
            label="covtype",
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
    def __init__(self, n_qubits, n_layers=6):
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
        # Bottleneck the classical transformation to prevent feature explosion
        hidden_dim = min(input_dim, n_qubits * 2)
        self.clayer_in = nn.Linear(input_dim, hidden_dim)
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
        "digits": "Digits",
        "covtype": "Covtype",
        "circles": "Circles",
    }.get(dataset_key, dataset_key)

def build_dataset_benchmarks(completed_labels=None, output_dir=OUTPUT_DIR):
    completed_labels = completed_labels or {}
    digit_label = completed_labels.get("digits", _digit_label(DIGIT_CLASSES))
    return {
        "Breast Cancer": {
            "a": _cancer_history_path(output_dir, "02_Output_BasicHistory.csv"),
            "b": _cancer_history_path(output_dir, "02_Output_ClassicalHistory.csv"),
            "c": _cancer_history_path(output_dir, "02_Output_QNNHistory.csv"),
            "d": _cancer_history_path(output_dir, "02_Output_StrongHistory.csv")
        },
        "Digits": {
            "a": _history_path(output_dir, "Basic_VQC", digit_label),
            "b": _history_path(output_dir, "Classical_DNN", digit_label),
            "c": _history_path(output_dir, "Hybrid_QNN", digit_label),
            "d": _history_path(output_dir, "Strong_VQC", digit_label)
        },
        "Covtype": {
            "a": _history_path(output_dir, "Basic_VQC", "covtype"),
            "b": _history_path(output_dir, "Classical_DNN", "covtype"),
            "c": _history_path(output_dir, "Hybrid_QNN", "covtype"),
            "d": _history_path(output_dir, "Strong_VQC", "covtype")
        },
        "Circles": {
            "a": _history_path(output_dir, "Basic_VQC", "circles"),
            "b": _history_path(output_dir, "Classical_DNN", "circles"),
            "c": _history_path(output_dir, "Hybrid_QNN", "circles"),
            "d": _history_path(output_dir, "Strong_VQC", "circles")
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
