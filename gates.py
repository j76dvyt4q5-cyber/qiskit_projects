import numpy as np
from qiskit.quantum_info import Operator

#Identity Gate
I = Operator([[1, 0],
             [0, 1]])

#Pauli-X Gate
X = Operator([[0, 1],
             [1, 0]])

#Pauli-Z Gate
Z = Operator([[1, 0],
             [0, -1]])  

#Pauli-Y Gate
Y = Operator([[0, -1j],
             [1j, 0]])

#Hadamard Gate
H = Operator([[1/np.sqrt(2), 1/np.sqrt(2)],
             [1/np.sqrt(2), -1/np.sqrt(2)]])
