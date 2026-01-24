from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile
from gates import I, X, Y, Z, H, P, S, T
from states import ket0, ket1, ket00, ket01, ket10, ket11

#CCNOT Gate Circuit on 3 qubits, 1 swapped and 1 superimposed
#Creates a X(0)*H(1)*CCX(0,1,2) state
qc = QuantumCircuit(3)
qc.x(0)
qc.h(1)
qc.ccx(0, 1, 2)

qc.measure_all()
qc.draw(output="text")

sim = AerSimulator()
t_qc = transpile(qc, sim)
result = sim.run(t_qc).result()
counts = result.get_counts()
print(counts)

#Rotation Gate Circuit
qc = QuantumCircuit(2)

qc.ry(np.pi, 1) #Rotates q1 to |1>
qc.rz(np.pi/2, 0) #Rotates q0 to superposition state

qc.h(0) #Creates interferance, revealing rotation gates
qc.measure_all()
qc.draw(output="text")

sim = AerSimulator()
t_qc = transpile(qc, sim)
result = sim.run(t_qc).result()
counts = result.get_counts()
print(counts)

#Classical Baseline
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

#Actual Quantum Interference Baseline
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

#Inteferance Test
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.h(0)
qc.measure_all()
qc.draw(output="text")
sim = AerSimulator()
t_qc = transpile(qc, sim)
result = sim.run(t_qc).result()
counts = result.get_counts()
print(counts)

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.measure(0, 0)
qc.cx(0, 1)
qc.h(0)
qc.measure(1, 1)
qc.draw(output="text")
sim = AerSimulator()
t_qc = transpile(qc, sim)
result = sim.run(t_qc).result()
counts = result.get_counts()
print(counts)

#Phase Test
qc = QuantumCircuit(1)
qc.h(0)
qc.t(0)
qc.h(0)
qc.measure_all()
qc.draw(output="text")
sim = AerSimulator()
t_qc = transpile(qc, sim)
result = sim.run(t_qc).result()
counts = result.get_counts()
print(counts)
