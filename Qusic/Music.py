from midiutil import MIDIFile
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.quantum_info import Statevector
import os

app = Flask(__name__)

# ── CORS: allows the browser frontend to talk to this server ──────────────────
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# ── QUANTUM CIRCUIT ───────────────────────────────────────────────────────────
# 10 qubits: 0-4 = melody register, 5-9 = accompaniment register
# Each register has 5 qubits = 32 possible states = 32 notes
qc = QuantumCircuit(10)
for i in range(10):
    qc.h(i)  # start in equal superposition — all notes equally likely

# ── GATE MAP: maps string names to Qiskit gate methods ───────────────────────
gate_map = {
    'H': qc.h,
    'X': qc.x,
    'Z': qc.z,
    'RY': qc.ry,
    'RZ': qc.rz,
}

# ── NOTE MAPS: generated programmatically ────────────────────────────────────
# melody: C4 (60) through 32 chromatic semitones up
melody_map = {}
for i in range(32):
    bitstring = format(i, '05b')  # 5-bit binary string e.g. '00000', '00001'
    melody_map[bitstring] = 60 + i  # C4 = 60, C#4 = 61, etc.

# accompaniment: C2 (36) through 32 chromatic semitones up
accompaniment_map = {}
for i in range(32):
    bitstring = format(i, '05b')
    accompaniment_map[bitstring] = 36 + i  # C2 = 36

# ── NOTE NAMES: for display in the UI ────────────────────────────────────────
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_to_name(midi):
    # converts MIDI number to note name e.g. 60 → "C4"
    octave = (midi // 12) - 1
    note = NOTE_NAMES[midi % 12]
    return f"{note}{octave}"

# ── MEASURE FUNCTION ──────────────────────────────────────────────────────────
def measure_circuit():
    # make a copy so we don't collapse the original circuit
    qc_copy = qc.copy()
    qc_copy.measure_all()
    sim = AerSimulator(shots=32)
    t_qc = transpile(qc_copy, sim)
    result = sim.run(t_qc).result()
    counts = result.get_counts()
    print(counts)
    return counts

# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    # serves the HTML file
    return send_from_directory('.', 'qusic.html')

@app.route('/apply_gate', methods=['POST'])
def apply_gate():
    # receives gate name, qubit, and optional angle from the frontend
    data = request.get_json()
    gate = data['gate']
    qubit = data['qubit']
    angle = data.get('angle', None)  # .get() returns None if key doesn't exist

    if gate in ['RY', 'RZ']:
        # rotation gates need an angle as first argument
        gate_map[gate](float(angle), qubit)
    else:
        gate_map[gate](qubit)

    return jsonify({'status': 'ok', 'gate': gate, 'qubit': qubit})

@app.route('/reset', methods=['GET'])
def reset():
    # wipes the circuit back to fresh equal superposition
    global qc  # need global here because we're reassigning qc, not just modifying it
    qc = QuantumCircuit(10)
    for i in range(10):
        qc.h(i)
    # also update gate_map to point to new circuit
    gate_map['H'] = qc.h
    gate_map['X'] = qc.x
    gate_map['Z'] = qc.z
    gate_map['RY'] = qc.ry
    gate_map['RZ'] = qc.rz
    return jsonify({'status': 'ok'})

@app.route('/ghz', methods=['GET'])
def ghz():
    # applies 6-qubit GHZ state — maximum entanglement across both registers
    # result: melody and accompaniment always collapse to same extreme (all low or all high)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 4)
    qc.cx(4, 5)
    qc.cx(5, 6)
    qc.cx(6, 7)
    qc.cx(7, 8)
    qc.cx(8, 9)
    return jsonify({'status': 'ok'})

@app.route('/probabilities', methods=['GET'])
def probabilities():
    # returns current quantum state probabilities WITHOUT measuring/collapsing
    # this is what powers the real-time display
    sv = Statevector(qc)
    probs = sv.probabilities_dict()

    # split into melody and accompaniment
    melody_probs = {}
    acc_probs = {}

    for bitstring, prob in probs.items():
        if prob < 0.0001:
            continue  # skip near-zero probabilities
        melody_bits = bitstring[5:]
        acc_bits = bitstring[:5]
        melody_note = midi_to_name(melody_map[melody_bits])
        acc_note = midi_to_name(accompaniment_map[acc_bits])

        # accumulate probabilities per note
        melody_probs[melody_note] = melody_probs.get(melody_note, 0) + prob
        acc_probs[acc_note] = acc_probs.get(acc_note, 0) + prob

    # sort by probability descending
    melody_probs = dict(sorted(melody_probs.items(), key=lambda x: -x[1]))
    acc_probs = dict(sorted(acc_probs.items(), key=lambda x: -x[1]))

    return jsonify({'melody': melody_probs, 'accompaniment': acc_probs})

@app.route('/generate', methods=['GET'])
def generate():
    print("generate route hit")
    counts = measure_circuit()
    melody_sequence = []
    accompaniment_sequence = []

    for bitstring, count in counts.items():
        melody_bits = bitstring[:5]
        accompaniment_bits = bitstring[5:]
        melody_sequence.extend([melody_map[melody_bits]] * count)
        accompaniment_sequence.extend([accompaniment_map[accompaniment_bits]] * count)

    print(f"Melody: {melody_sequence}")
    print(f"Accompaniment: {accompaniment_sequence}")

    midi = MIDIFile(2)
    midi.addTempo(0, 0, 120)
    midi.addTempo(1, 0, 120)

    for i, pitch in enumerate(melody_sequence):
        midi.addNote(0, 0, pitch, i, 1, 80)
    for i, pitch in enumerate(accompaniment_sequence):
        midi.addNote(1, 1, pitch, i, 1, 60)

    output_path = os.path.join(os.path.dirname(__file__), 'quantum_melody_acc.mid')
    with open(output_path, "wb") as f:
        midi.writeFile(f)

    return send_from_directory(os.path.dirname(__file__), 'quantum_melody_acc.mid', as_attachment=True)

# ── START SERVER ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\n Qusic server starting...')
    print(' Open http://localhost:5000 in your browser.\n')
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)