import numpy as np
from qiskit.quantum_info import Statevector, Operator

def show(label, state):
    print(f"\n{label}:")
    print(state.draw("text"))

ket0 = Statevector([1, 0])
ket1 = Statevector([0, 1])

show("ket0", ket0)
show("ket1", ket1)

