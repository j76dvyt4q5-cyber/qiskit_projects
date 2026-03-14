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

while not generate_song:
    superposition_action = input("Would you like to put your melody or accompaniment into superposition? {1. Melody | 2. Accompaniment}")
    if superposition_action == "1":
        qc.h(0)
        qc.h(1)
        qc.h(2)
    if superposition_action == "2":
        qc.h(3)
        qc.h(4)
        qc.h(5)
    melody_action = input("What gates would you like to apply to your melody? {1. H | 2. X | 3. Z | 4. CX | 5. CY | 6. CZ | 7. T}")
    #H GATE
    if melody_action == "1":
        melody_action_H_choice = input("Which qubit would you like to apply an H gate to? {1. 0 | 2. 1 | 3. 2}")
        qc.h(int(melody_action_H_choice) - 1)

    #X GATE
    if melody_action == "2":
        melody_action_X_choice = input("Which qubit would you like to apply an X gate to? {1. 0 | 2. 1 | 3. 2}")
        qc.x(int(melody_action_X_choice) - 1)

    #Z GATE
    if melody_action == "3":
        melody_action_Z_choice = input("Which qubit would you like to apply an Z gate to? {1. 0 | 2. 1 | 3. 2}")
        qc.z(int(melody_action_Z_choice) - 1)

    #CX GATE
    if melody_action == "4":
        melody_action_CX_choice = input("Which two qubits would you like to entangle? {1. 0 & 1 | 2. 0 & 2 | 3. 1 & 2}")
        if melody_action_CX_choice == "1":
            CX_choice_1 = input("Choose target and control qubits {1. Target: 0, Control: 1 | 2. Target: 1, Control: 0}")
            if CX_choice_1 == "1":
                qc.cx(0, 1)
            if CX_choice_1 == "2":
                qc.cx(1, 0)
        if melody_action_CX_choice == "2":
            CX_choice_2 = input("Choose target and control qubits {1. Target: 0, Control: 2 | 2. Target: 2, Control: 0}")
            if CX_choice_2 == "1":
                qc.cx(0, 2)
            if CX_choice_2 == "2":
                qc.cx(2, 0)
        if melody_action_CX_choice == "3":
            CX_choice_3 = input("Choose target and control qubits {1. Target: 1, Control: 2 | 2. Target: 2, Control: 1}")
            if CX_choice_3 == "1":
                qc.cx(0, 2)
            if CX_choice_3 == "2":
                qc.cx(2, 0)
    #CY GATE
    if melody_action == "5":
        melody_action_CY_choice = input("Which two qubits would you like to apply a CY gate to? {1. 0 & 1 | 2. 0 & 2 | 3. 1 & 2}")
        if melody_action_CY_choice == "1":
            CY_choice_1 = input("Choose target and control qubits {1. Target: 0, Control: 1 | 2. Target: 1, Control: 0}")
            if CY_choice_1 == "1":
                qc.cy(0, 1)
            if CY_choice_1 == "2":
                qc.cy(1, 0)
        if melody_action_CY_choice == "2":
            CY_choice_2 = input("Choose target and control qubits {1. Target: 0, Control: 2 | 2. Target: 2, Control: 0}")
            if CY_choice_2 == "1":
                qc.cy(0, 2)
            if CY_choice_2 == "2":
                qc.cy(2, 0)
        if melody_action_CY_choice == "3":
            CY_choice_3 = input("Choose target and control qubits {1. Target: 1, Control: 2 | 2. Target: 2, Control: 1}")
            if CY_choice_3 == "1":
                qc.cy(0, 2)
            if CY_choice_3 == "2":
                qc.cy(2, 0)
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
