from backend import simulate_circuit

result = simulate_circuit(
    n_qubits=1,
    init="0",
    gates=[],
    shots=1024
)

print(result)