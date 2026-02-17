#Inteferance Test
from qiskit import transpile
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile
from gates import I, X, Y, Z, H, P, S, T
from states import ket0, ket1, ket00, ket01, ket10, ket11


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

def t_1():
    if counts == {"00": 1024}:
        print("Test passed!")
    else:
        print("Test failed. Expected {'00': 1024}, got", counts)
t_1()

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

def t_2():
    if counts == {"00": 512, "11": 512}: #write code for something that can 
        #detect outputs within a certain range
        print("Test passed!")
    else: 
        print("Test failed. Expected {'00': 512, '11': 512}, got", counts)
t_2()