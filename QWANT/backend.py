from fastapi import FastAPI
from qiskit import QuantumCircuit

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


def simulate_circuit(n_qubits, init, gates, shots=None):
    """
    Build a circuit from a basis-state init string (e.g., "01") and return a
    text drawing of the circuit.

    Notes:
    - init[i] corresponds to qubit i (Qiskit qubit index).
    - For now, gates are ignored. We'll add them next step.
    """

    # --------
    # Validation
    # --------
    if not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError("n_qubits must be an integer >= 1")

    if not isinstance(init, str):
        raise ValueError("init must be a string")

    if len(init) != n_qubits:
        raise ValueError(f"init must have length {n_qubits} (got {len(init)})")

    if not set(init) <= {"0", "1"}:
        raise ValueError("init must contain only '0' and '1'")

    if not isinstance(gates, list):
        raise ValueError("gates must be a list")

    if shots is not None:
        if not isinstance(shots, int) or shots < 1:
            raise ValueError("shots must be an integer >= 1 or None")

    # --------------------------
    # Build circuit from init
    # --------------------------
    qc = QuantumCircuit(n_qubits)

    for i, bit in enumerate(init):
        if bit == "1":
            qc.x(i)

    # Return a readable circuit drawing
    return {"circuit": str(qc.draw(output="text"))}

