from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile
from gates import I, X, Y, Z, H, P, S, T
from states import ket0, ket1, ket00, ket01, ket10, ket11

#Independent circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.measure_all()
qc.draw(output="text")

sim = AerSimulator()
t_qc = transpile(qc, sim)
result = sim.run(t_qc).result()
counts = result.get_counts()
print(counts)

#Entangled circuit (Bell State)
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()
qc.draw(output="text")

sim = AerSimulator()
t_qc = transpile(qc, sim)
result = sim.run(t_qc).result()
counts = result.get_counts()
print(counts)

#Broken entangled circuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.measure(0, 0)
qc.cx(0, 1)
qc.measure(1, 1)
qc.draw(output="text")

sim = AerSimulator()
t_qc = transpile(qc, sim)
result = sim.run(t_qc).result()
counts = result.get_counts()
print(counts)





