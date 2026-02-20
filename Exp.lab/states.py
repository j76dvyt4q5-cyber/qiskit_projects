from readline import backend
from readline import execute 
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
from qiskit_aer import AerSimulator

from qiskit_aer import Aer
from qiskit import transpile
from gates import I, X, Y, Z, H, P, S, T
from qiskit.visualization import array_to_latex
ket0 = Statevector([1,0])
ket1 = Statevector([0,1])

ket00 = Statevector([1,0,0,0])
ket01 = Statevector([0,1,0,0])
ket10 = Statevector([0,0,1,0])
ket11 = Statevector([0,0,0,1])

qc = QuantumCircuit(3)

qc.h(0)
qc.h(1)
qc.h(2)

out = execute(qc,backend).result().get_unitary()
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)
job = simulator.run(compiled_circuit, shots=1)
result = job.result()
statevector = result.get_statevector(qc)
array_to_latex(statevector, prefix="\text{Statevector= }")
