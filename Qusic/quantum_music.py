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
counts = measure_qc_1024()
#NOTES
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
melody_sequence = []
accompaniment_sequence = []
for bitstring, count in counts.items():
    melody_bits = bitstring[:3]
    accompaniment_bits = bitstring[3:]
    print(f"Melody:", melody_sequence)
    print("Accompaniment:", accompaniment_sequence)
    melody_sequence.extend([melody_map[melody_bits]] * count)
    accompaniment_sequence.extend([accompaniment_map[accompaniment_bits]] * count)

#MIDI FILE
midi = MIDIFile(2)
midi.addTempo(0, 0, 120)
for i, pitch in enumerate(melody_sequence):
    midi.addNote(0, 0, pitch, i, 1, 80)

for i, pitch in enumerate(accompaniment_sequence):
    midi.addNote(1, 1, pitch, i, 1, 60)

with open("quantum_melody.mid", "wb") as f:
    midi.writeFile(f)

print(f"Done! quantum_melody.mid1 created.")
