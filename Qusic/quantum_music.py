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
qc = QuantumCircuit(3)
qc.h(0)
qc.h(1)
qc.h(2)
qc.cx(0, 1)
qc.cx(1, 2)
qc.ry(np.pi/3, 0)
qc.ry(np.pi/6, 1)
qc.rz(np.pi/4, 2)
counts = measure_qc_1024()
#NOTES
note_map = {
    '000': 60,
    '001': 62,
    '010': 64,
    '011': 65,
    '100': 67,
    '101': 69,
    '110': 71,
    '111': 72,
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

print("Done! quantum_melody.mid created.")
