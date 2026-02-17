from qiskit import QuantumCircuit
from qiskit import *
from qiskit.quantum_info import Statevector
import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile
from gates import I, X, Y, Z, H, P, S, T
from states import ket0, ket1, ket00, ket01, ket10, ket11
from pylatexenc import matplotlib 
qc = QuantumCircuit(1)
qc.x(0)

qc.draw()
