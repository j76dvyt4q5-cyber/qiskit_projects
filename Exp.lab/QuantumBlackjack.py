from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile
from gates import I, X, Y, Z, H, P, S, T
from qiskit.visualization import array_to_latex
import math 
def measure_qc_1024():
    qc.measure_all()
    qc.draw(output="text")
    sim = AerSimulator()
    t_qc = transpile(qc, sim)
    result = sim.run(t_qc).result()
    counts = result.get_counts()
    print(counts)

def measure_qc_card_cout(qc):
    qc.measure(card, c_out)
    sim = AerSimulator()
    t_qc = transpile(qc, sim)
    result = sim.run(t_qc).result()
    counts = result.get_counts()
    print(counts)

def qc_variables(n):
    global qc_0, qc_1, qc_2
    if n == 1:
        qc_0 = 0
    elif n == 2:
        qc_0, qc_1 = 0, 1
    elif n == 3:
        qc_0, qc_1, qc_2 = 0, 1, 2
    elif n > 3:
        globals()[f"qc_{n-1}"] = n-1

def apply_table_gate(qc, control_table, table, card, target_hand, theta, table_card_gate):
    if table_card_gate == "CRY":
        qc.cry(theta, table[control_table], card[target_hand])
    elif table_card_gate == "CX":
        qc.cx(table[control_table], card[target_hand])
    elif table_card_gate == "CZ":
        qc.cz(table[control_table], card[target_hand])

def apply_self_gate(qc, card, self_gate, target_index, strength=None):
    if self_gate == "H":
        qc.h(card[target_index])
    elif self_gate == "Z":
        qc.z(card[target_index])
    elif self_gate == "X":
        qc.x(card[target_index])
    elif self_gate == "RY":
        if strength is None:
            raise ValueError("Strength must be provided for RY gate")
        qc.ry(strength, card[target_index])
    else:
        print("Invalid gate. Try again")

#Initialization of Table and Cards
card = QuantumRegister(4, "card")
table = QuantumRegister(2, "table")
c_out = ClassicalRegister(4, "c_out")
qc = QuantumCircuit(card, table, c_out) #Sample circuit
qc.h(card)
qc.h(table[1])

#Controlled Gate with Table and Card
control_table = int(input("Choose your control table bit (0-1): "))
target_hand = int(input("Choose your target hand bit (0-3): "))
table_card_gate = input("Choose the gate to apply between table and card (CRY, CX, CZ): ").strip().upper()
mode = input("Mode (h = help, s = sabotage): ").lower()
sign = +1 if mode == "h" else -1
strength = int(input("Strength (1-3): "))
theta_table = sign * [0.0, 0.4, 0.8, 1.2][strength]
apply_table_gate(qc, control_table, table, card, target_hand, theta_table, table_card_gate)

#Self Gate on Card
self_gate = input("Choose the gate to apply to your own hand(H, Z, X, RY): ").strip().upper()
target_self = int(input("Choose card qubit (0-3): "))
if self_gate == "RY":
    strength_level = int(input("Strength (1-3): "))
    theta_self = [0.0, 0.4, 0.8, 1.2][strength_level] 
else:
    theta_self = None

apply_self_gate(qc, card, self_gate, target_self, theta_self)

measure_qc_card_cout(qc)


        
