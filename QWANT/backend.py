from fastapi import FastAPI
from qiskit import QuantumCircuit

app = FastAPI()
@app.get("/health")

def health():
    return {"status": "ok"}

def simulate_circuit(n_qubits, init, gates, shots=None):
    return {"status": "ok"}