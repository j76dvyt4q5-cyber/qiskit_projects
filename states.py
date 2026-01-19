import numpy as np
from qiskit.quantum_info import Statevector

ket0 = Statevector([1,0])
ket1 = Statevector([0,1])

ket00 = Statevector([1,0,0,0])
ket01 = Statevector([0,1,0,0])
ket10 = Statevector([0,0,1,0])
ket11 = Statevector([0,0,0,1])
