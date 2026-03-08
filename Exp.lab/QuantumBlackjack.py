from unittest import result
import random
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

def measure_qc_card_cout(qc, card, c_out):
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

def quantum_index(card, table, c_out, table_mods=None, self_mods=None):
    qc = QuantumCircuit(card, table, c_out)
    qc.h(card)
    qc.h(table)
    if table_mods:
        for control_table, target_hand, table_card_gate, theta_table in table_mods:
            apply_table_gate(qc, control_table, table, card, target_hand, theta_table, table_card_gate)
    if self_mods:
        for self_gate, target_self, theta_self in self_mods:
            apply_self_gate(qc, card, self_gate, target_self, theta_self)
    counts = measure_qc_card_cout(qc, card, c_out)
    bitstring = list(counts.keys())[0]
    return int(bitstring, 2)

def fresh_deck():
    faces = ['Ace','2','3','4','5','6','7','8','9','10','Jack','Queen','King']
    deck = faces * 4
    random.shuffle(deck)
    return deck

def draw_card(deck, card, table, c_out, table_mods=None, self_mods=None):
    if not deck:
        deck.extend(fresh_deck())
        print(" [Deck reshuffled]")
    while True:
        idx = quantum_index(card, table, c_out, table_mods, self_mods)
        if idx < len(deck):
            drawn = deck[idx]
            deck.pop(idx)
            return drawn
        table_mods = None
        self_mods = None

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

#TABLE AND CARDS FOR STARTING HAND
deck = fresh_deck()
player_total = 0
ace_count = 0
standing = False
hand_cards = []
table_modifiers = []
self_modifiers = []
card = QuantumRegister(4, "card")
table = QuantumRegister(2, "table")
c_out = ClassicalRegister(4, "c_out")
qc = QuantumCircuit(card, table, c_out) 
qc.h(card)
qc.h(table)

def set_up(player_total, ace_count):
    card_drawn_0 = draw_card(deck, card, table, c_out)
    player_total, ace_count = add_card_to_total(card_drawn_0, player_total, ace_count)

    card_drawn_1 = draw_card(deck, card, table, c_out)
    player_total, ace_count = add_card_to_total(card_drawn_1, player_total, ace_count)

    print(f"You start with a {card_drawn_0} and a {card_drawn_1}, with a total of {player_total}.")
    return player_total, ace_count, card_drawn_0, card_drawn_1

player_total, ace_count, start0, start1 = set_up(player_total, ace_count)
hand_cards.extend([start0, start1])


while (player_total <= 21) and (not standing):
    action = input(
    "Do you want to hit, apply a gate to your hand, apply a gate between table and card, use your ultimate, or stand? (hit/self/table/stand/ultimate): "
).strip().lower()
    if action == "hit":
        card_drawn = draw_card(deck, card, table, c_out, table_modifiers, self_modifiers)
        hand_cards.append(card_drawn)
        player_total, ace_count = add_card_to_total(card_drawn, player_total, ace_count)
        if player_total > 21:
            print(f"Bust! Total: {player_total}, Drew: {card_drawn}, Hand: {', '.join(hand_cards)}")
            standing = True
        else:
            print(f"You drew a {card_drawn}. Hand: {', '.join(hand_cards)} | Total: {player_total}")
        table_modifiers.clear()
        self_modifiers.clear()

    elif action == "self":
        self_gate = input("Choose the gate to apply to your own hand(H, Z, X, RY): ").strip().upper()
        target_self = int(input("Choose card qubit (0-3): "))
        if self_gate == "RY":
            strength_level = int(input("Strength (1-3): "))
            theta_self = [0.0, 0.4, 0.8, 1.2][strength_level] 
        else:
            theta_self = None
        self_modifiers.append((self_gate, target_self, theta_self))
        print(f"Applied {self_gate} gate to card bit {target_self} with strength {theta_self if theta_self is not None else 'N/A'}")
        print("Self modifier saved. It will affect the next hit.")

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
        table_modifiers.append((control_table, target_hand, table_card_gate, theta_table))
        print(f"Table modifier saved. It will affect the next hit.")
        

    elif action == "ultimate":
        table_modifiers.append((0, 0, "CX", None))
        table_modifiers.append((1, 1, "CX", None))
        print("Used ultimate: will apply CX gates between table bits and their corresponding card bits on the next hit.")

    elif action == "stand":
        standing = True
        print("Stand. No more actions taken.")

    else:
    #Invalid
        print("Invalid action. No draw.")


#DEALER
if standing and player_total <= 21:
    dealer_total = 0
    dealer_ace_count = 0
    dealer_hand = []

    while dealer_total < 17:
        #New dealer circuit
        qc = QuantumCircuit(card, table, c_out)
        qc.h(card)
        qc.h(table)
        table_modifiers.clear()
        counts_dealer = measure_qc_card_cout(qc, card, c_out)
        card_drawn_dealer = draw_card(deck, card, table, c_out)

        dealer_hand.append(card_drawn_dealer)
        dealer_total, dealer_ace_count = add_card_to_total(
            card_drawn_dealer,
            dealer_total,
            dealer_ace_count
        )

        print(f"Dealer drew a {card_drawn_dealer}. Dealer total: {dealer_total}")

    print("Dealer hand:", ", ".join(dealer_hand), "| Total:", dealer_total)

    if dealer_total > 21:
        print("Dealer busts! You win!")
    elif dealer_total > player_total:
        print("Dealer wins with a total of", dealer_total)
    elif dealer_total < player_total:
        print("You win with a total of", player_total)
    else:
        print("It's a tie with both at", player_total)




        

