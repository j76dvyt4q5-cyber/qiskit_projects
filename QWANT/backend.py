from fastapi import FastAPI
from qiskit import QuantumCircuit

app = FastAPI()
@app.get("/health")

def health():
    return {"status": "ok"}

def simulate_circuit(n_qubits, init, gates, shots=None):
    if not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError("n_qubits must be an integer >= 1")

    if not isinstance(init, str):
        raise ValueError("init must be a string")

    if len(init) != n_qubits:
        raise ValueError(f"init must have length {n_qubits} (got {len(init)})")

    if not set(init) <= {"0", "1"}:
        raise ValueError("init must contain only '0' and '1'")

    return {"status": "ok"}
