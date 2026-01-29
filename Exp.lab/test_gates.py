import numpy as np
from qiskit.quantum_info import Statevector
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

ket0 = Statevector([1,0])
ket1 = Statevector([0,1])
gates = [("I", I), ("X", X), ("Y", Y), ("Z", Z), ("H", H)]
input_states = [("|0>", ket0), ("|1>", ket1)]


#fix
for name, gate in gates:
    for label, ket in input_states:
        state = gate @ ket
        probs = state.probabilities_dict()
        for outcome, prob in probs.items():
            line = f"{name} {label} = {outcome}"
            print(f"{line} {float(prob):.3f}")




