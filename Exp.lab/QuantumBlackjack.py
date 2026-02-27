from unittest import result
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
    result = sim.run(t_qc, shots=1).result()
    counts = result.get_counts()
    return(counts)

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

def bit_to_card(counts):
    bitstring = list(counts.keys())[0]
    card_mapping = {
        '0000': 'Unknown',
        '0001': 'Ace',
        '0010': '2',
        '0011': '3',
        '0100': '4',
        '0101': '5',
        '0110': '6',
        '0111': '7',
        '1000': '8',
        '1001': '9',
        '1010': '10',
        '1011': 'Jack',
        '1100': 'Queen',
        '1101': 'King',
        '1110': 'Ace',
        '1111': 'Unknown'
    }
    card_value = card_mapping.get(bitstring, 'Unknown')
    return card_value

def card_to_value(card):
    card = card.strip()
    if card in ['Jack', 'Queen', 'King']:
        return 10
    elif card == 'Ace':
        return 11
    if card == 'Unknown':
        return 0
    if card.isdigit():
        return int(card)
    return 0

def add_card_to_total(card, player_total, ace_count):
    if card == 'Ace':
        player_total += 11
        ace_count += 1
    else:
        player_total += card_to_value(card)
    while player_total > 21 and ace_count > 0:
        player_total -= 10
        ace_count -= 1
    return player_total, ace_count
    

#Initialization of Table and Cards
player_total = 0
ace_count = 0
standing = False
card = QuantumRegister(4, "card")
table = QuantumRegister(2, "table")
c_out = ClassicalRegister(4, "c_out")
qc = QuantumCircuit(card, table, c_out) 
qc.h(card)
qc.h(table)

while (player_total <= 21) and (not standing):
    action = input(
    "Do you want to hit, apply a gate to your hand, apply a gate between table and card, use your ultimate, or stand? (hit/self/table/stand/ultimate): "
).strip().lower()
    if action == "hit":
        counts = measure_qc_card_cout(qc)  # make sure this returns counts (shots=1 inside)
        card_drawn = bit_to_card(counts)
        player_total, ace_count = add_card_to_total(card_drawn, player_total, ace_count)

    elif action == "self":
        self_gate = input("Choose the gate to apply to your own hand(H, Z, X, RY): ").strip().upper()
        target_self = int(input("Choose card qubit (0-3): "))
        if self_gate == "RY":
            strength_level = int(input("Strength (1-3): "))
            theta_self = [0.0, 0.4, 0.8, 1.2][strength_level] 
        else:
            theta_self = None
        apply_self_gate(qc, card, self_gate, target_self, theta_self)

    elif action == "table":
        control_table = int(input("Choose your control table bit (0-1): "))
        target_hand = int(input("Choose your target hand bit (0-3): "))
        table_card_gate = input("Choose the gate to apply between table and card (CRY, CX, CZ): ").strip().upper()
        if table_card_gate == "CRY":
            mode = input("Mode (h = help, s = sabotage): ").lower()
            sign = +1 if mode == "h" else -1
            strength = int(input("Strength (1-3): "))
            theta_table = sign * [0.0, 0.4, 0.8, 1.2][strength]
        else:
            theta_table = None
        apply_table_gate(qc, control_table, table, card, target_hand, theta_table, table_card_gate)

    elif action == "ultimate":
        qc.cx(table[0], card[0])
        qc.cx(table[1], card[1])

    elif action == "stand":
        standing = True
        print("Stand. No more actions taken.")
    else:
    # invalid
        print("Invalid action. No draw.")

if player_total > 21:
    print(f"Bust! Your card total is {player_total}, Counts is {counts}, You drew a {card}")
else:
    print(f"Your player total is {player_total}, Counts is {counts}, You drew a {card}")



        
