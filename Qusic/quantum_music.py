from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import AerSimulator
from midiutil import MIDIFile
import numpy as np

def measure_qc_1024():
    qc.measure_all()
    qc.draw(output="text")
    sim = AerSimulator(shots=32)
    t_qc = transpile(qc, sim)
    result = sim.run(t_qc).result()
    counts = result.get_counts()
    print(counts)
    return counts

#QUANTUM CIRCUIT
generate_song = False
melody_sequence = []
accompaniment_sequence = []
qc = QuantumCircuit(6)
qc.h(0)
qc.h(1)
qc.h(2)
qc.h(3)
qc.h(4)
qc.h(5)
qc.cx(0, 3)
qc.cx(1, 4)
qc.cx(2, 5)
qc.ry(np.pi/4, 0)
qc.ry(-np.pi/4, 1)
qc.ry(np.pi/4, 2)
qc.rz(-np.pi/6, 3)
qc.rz(np.pi/3, 4)
qc.rz(-np.pi/4, 5)

#MAP DICTS
gate_map = {'1': qc.h,'2': qc.x,'3': qc.z}

while not generate_song:
    action  = input("What would you like to do? {1. Superposition | 2. Apply gate | 3. Generate}")
    if action == "3":
        generate_song = True
        continue
    if action == "1":
        superposition_action = input("Would you like to put your melody or accompaniment into superposition? {1. Melody | 2. Accompaniment}")
        if superposition_action == "1":
            qc.h(0)
            qc.h(1)
            qc.h(2)
        if superposition_action == "2":
            qc.h(3)
            qc.h(4)
            qc.h(5)
    if action == "2":
        action_choice = input("Would you like to apply gates to your melody or accompaniment? {1. M | 2. A}")
        if action_choice == "1":
            melody_action = input("What gates would you like to apply to your melody? {1. H | 2. X | 3. Z | 4. CX | 5. CY | 6. CZ | 7. T}")
            double_gate_map = {'1':(0, 1), '2':(0, 2),'3':(1, 2)}
            double_gate_choice_1 = {'1':(0, 1),'2':(1, 0)}
            double_gate_choice_2 = {'1':(0, 2),'2':(2, 0)}
            double_gate_choice_3 = {'1':(1, 2),'2':(2, 1)}
            triple_gate_choice = {'1':(0, 1, 2), '2':(0, 2, 1), '3':(1, 2, 0)}
        if action_choice == '2':
            double_gate_map = {'1':(3, 4), '2':(3, 5), '3':(4, 5)}
            double_gate_choice_1 = {'1':(3, 4),'2':(4, 3)}
            double_gate_choice_2 = {'1':(3, 5),'2':(5, 3)}
            double_gate_choice_3 = {'1':(4, 5),'2':(5, 4)}
            triple_gate_choice = {'1':(3, 4, 5), '2':(3, 5, 4), '3':(4, 5, 3)}

    #H, X, Z GATES
        if melody_action in gate_map:
            single_melody_qubit = int(input("Which qubit would you like to target? { 1. 0 | 2. 1 | 3. 2}"))
            gate_map[melody_action](single_melody_qubit)
    #CX GATE
        if melody_action == "4":
            melody_action_CX_choice = input("Which two qubits would you like to entangle? {1. 0 & 1 | 2. 0 & 2 | 3. 1 & 2}")
            if melody_action_CX_choice == "1":
                CX_choice_1 = input("Choose target and control qubits {1. Target: 0, Control: 1 | 2. Target: 1, Control: 0}")
                CX_pair = double_gate_map[melody_action_CX_choice]
                CX_order = double_gate_choice_1[CX_choice_1]
                qc.cx(*CX_order)
            if melody_action_CX_choice == "2":
                CX_choice_2 = input("Choose target and control qubits {1. Target: 0, Control: 2 | 2. Target: 2, Control: 0}")
                CX_pair = double_gate_map[melody_action_CX_choice]
                CX_order = double_gate_choice_1[CX_choice_1]
                qc.cx(*CX_order)
            if melody_action_CX_choice == "3":
                CX_choice_3 = input("Choose target and control qubits {1. Target: 1, Control: 2 | 2. Target: 2, Control: 1}")
                CX_pair = double_gate_map[melody_action_CX_choice]
                CX_order = double_gate_choice_1[CX_choice_1]
                qc.cx(*CX_order)

    #CY GATE
        if melody_action == "5":
            melody_action_CY_choice = input("Which two qubits would you like to apply a CY gate to? {1. 0 & 1 | 2. 0 & 2 | 3. 1 & 2}")
            if melody_action_CY_choice == "1":
                CY_choice_1 = input("Choose target and control qubits {1. Target: 0, Control: 1 | 2. Target: 1, Control: 0}")
                CY_pair = double_gate_map[melody_action_CY_choice]
                CY_order = double_gate_choice_1[CY_choice_1]
                qc.cy(*CY_order)
            if melody_action_CY_choice == "2":
                CY_choice_2 = input("Choose target and control qubits {1. Target: 0, Control: 2 | 2. Target: 2, Control: 0}")
                CY_pair = double_gate_map[melody_action_CY_choice]
                CY_order = double_gate_choice_1[CY_choice_1]
                qc.cy(*CY_order)
            if melody_action_CY_choice == "3":
                CY_choice_3 = input("Choose target and control qubits {1. Target: 1, Control: 2 | 2. Target: 2, Control: 1}")
                CY_pair = double_gate_map[melody_action_CY_choice]
                CY_order = double_gate_choice_1[CY_choice_1]
                qc.cy(*CY_order)
    #CZ GATE
        if melody_action == "6":
            melody_action_CZ_choice = input("Which two qubits would you like to apply a CZ gate to? {1. 0 & 1 | 2. 0 & 2 | 3. 1 & 2}")
            if melody_action_CZ_choice == "1":
                CZ_choice_1 = input("Choose target and control qubits {1. Target: 0, Control: 1 | 2. Target: 1, Control: 0}")
                CZ_pair = double_gate_map[melody_action_CZ_choice]
                CZ_order = double_gate_choice_1[CZ_choice_1]
                qc.cz(*CZ_order)
            if melody_action_CZ_choice == "2":
                CZ_choice_1 = input("Choose target and control qubits {1. Target: 0, Control: 1 | 2. Target: 1, Control: 0}")
                CZ_pair = double_gate_map[melody_action_CZ_choice]
                CZ_order = double_gate_choice_1[CZ_choice_1]
                qc.cz(*CZ_order)
            if melody_action_CZ_choice == "3":
                CZ_choice_1 = input("Choose target and control qubits {1. Target: 0, Control: 1 | 2. Target: 1, Control: 0}")
                CZ_pair = double_gate_map[melody_action_CZ_choice]
                CZ_order = double_gate_choice_1[CZ_choice_1]
                qc.cz(*CZ_order)

    #TOFFOLI
        if melody_action == "7":
            melody_action_T_choice = input("Choose target and control qubits {" \
            "1. Control: 0 & 1, Target: 2 | " \
            "2. Control: 0 & 2, Target: 1 | " \
            "3. Control: 1 & 2, Target: 0 | " \
            "}")
            T_pair = triple_gate_choice[melody_action_T_choice]
            qc.ccx(*T_pair)


counts = measure_qc_1024()
melody_map = {
    '000': 60,  # C4
    '001': 61,  # C#4
    '010': 62,  # D4
    '011': 63,  # D#4
    '100': 64,  # E4
    '101': 65,  # F4
    '110': 66,  # F#4
    '111': 67,  # G4
}

accompaniment_map = {
    '000': 48,  # C3
    '001': 49,  # C#3
    '010': 50,  # D3
    '011': 51,  # D#3
    '100': 52,  # E3
    '101': 53,  # F3
    '110': 54,  # F#3
    '111': 55,  # G3
}
#NOTES

note_sequence = []
for bitstring, count in counts.items():
    melody_bits = bitstring[:3]
    accompaniment_bits = bitstring[3:]
    melody_sequence.extend([melody_map[melody_bits]] * count)
    accompaniment_sequence.extend([accompaniment_map[accompaniment_bits]] * count)
print(f"Melody:", melody_sequence)
print("Accompaniment:", accompaniment_sequence)

#MIDI FILE
midi = MIDIFile(2)
midi.addTempo(0, 0, 120)
for i, pitch in enumerate(melody_sequence):
    midi.addNote(0, 0, pitch, i, 1, 80)

for i, pitch in enumerate(accompaniment_sequence):
    midi.addNote(1, 1, pitch, i, 1, 60)

with open("quantum_melody_acc.mid", "wb") as f:
    midi.writeFile(f)

print(f"Done! quantum_melody_acc.mid created.")
