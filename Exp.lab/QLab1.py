from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile
from gates import I, X, Y, Z, H, P, S, T
from qiskit.visualization import array_to_latex
def measure_qc_1024():
    qc.measure_all()
    qc.draw(output="text")
    sim = AerSimulator()
    t_qc = transpile(qc, sim)
    result = sim.run(t_qc).result()
    counts = result.get_counts()
    print(counts)


#CCNOT Gate Circuit on 3 qubits, 1 swapped and 1 superimposed
#Creates a X(0)*H(1)*CCX(0,1,2) state
qc = QuantumCircuit(3)
qc.x(0)
qc.h(1)
qc.ccx(0, 1, 2)

measure_qc_1024()

#Rotation Gate Circuit
qc = QuantumCircuit(2)

#qc.ry(np.pi, 1) #Rotates q1 to |1>
qc.rz(np.pi/4, 0) #Rotates q0 to hidden state
#qc.rx(np.pi/4, 0) #Further rotates q0
qc.h(0) #Creates interferance, converting phase + rotation into measurable bias
measure_qc_1024()

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
measure_qc_1024()

#Inteferance Test
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.h(0)
measure_qc_1024()
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
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 0)
measure_qc_1024()

#Other Tests
qc = QuantumCircuit(2)
qc.x(1)
qc.cy(1, 0)
qc.h(0)
measure_qc_1024()


qc = QuantumCircuit(2)
qc.h(0) #or x gate
qc.cx(0, 1)
qc.cx(1, 0)
qc.cx(0, 1)
measure_qc_1024()


qc = QuantumCircuit(2)
qc.h(0)
qc.swap(0, 1)
measure_qc_1024()


#Toffoli Gate
qc = QuantumCircuit(3)
q_0 = 0
q_1 = 1
q_2 = 2
qc.h((q_0, q_1, q_2))
qc.ccx(q_0, q_1, q_2)
measure_qc_1024()

#Toffoli Gate Identity
qc = QuantumCircuit(3)
q_0 = 0
q_1 = 1
q_2 = 2
qc.h((q_0, q_1, q_2))
qc.ch(q_0, q_2)
qc.cz(q_1, q_2)
qc.ch(q_0, q_2)
measure_qc_1024()
