from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import AerSimulator
from midiutil import MIDIFile
import numpy as np

def measure_qc_1024():
    qc.measure_all()
    qc.draw(output="text")
    sim = AerSimulator(shots=10)
    t_qc = transpile(qc, sim)
    result = sim.run(t_qc).result()
    counts = result.get_counts()
    print(counts)
    return counts

#QUANTUM CIRCUIT
qc = QuantumCircuit(4)
qc.h(0)
qc.h(1)
qc.h(2)
qc.h(3)
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)
qc.ry(np.pi/3, 0)
qc.ry(np.pi/6, 1)
qc.rz(np.pi/4, 2)
qc.rz(np.pi/4, 3)
counts = measure_qc_1024()
#NOTES
note_map = {
    '0000': 48,  # C3 (low)
    '0001': 50,  # D3
    '0010': 52,  # E3
    '0011': 60,  # C4 - chromatic starts here
    '0100': 61,  # C#4
    '0101': 62,  # D4
    '0110': 63,  # D#4
    '0111': 64,  # E4
    '1000': 65,  # F4
    '1001': 66,  # F#4
    '1010': 67,  # G4
    '1011': 68,  # G#4
    '1100': 69,  # A4
    '1101': 70,  # A#4
    '1110': 71,  # B4
    '1111': 72,  # C5 (high)
}
note_sequence = []
for bitstring, count in counts.items():
    note = note_map[bitstring]
    note_sequence.extend([note] * count)

#MIDI FILE
midi = MIDIFile(1)
midi.addTempo(0, 0, 120)
for i, pitch in enumerate(note_sequence):
    midi.addNote(0, 0, pitch, i, 1, 80)
with open("quantum_melody.mid", "wb") as f:
    midi.writeFile(f)

print(f"Done! quantum_melody.mid1 created.")
