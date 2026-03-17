from midiutil import MIDIFile
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile
import os

def measure_qc_1024():
    qc.measure_all()
    qc.draw(output="text")
    sim = AerSimulator(shots=32)
    t_qc = transpile(qc, sim)
    result = sim.run(t_qc).result()
    counts = result.get_counts()
    print(counts)
    return counts

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

app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/')
def home():
    return send_from_directory('.', 'qusic.html')

qc = QuantumCircuit(6)
gate_map = {'H': qc.h, 'X': qc.x, 'Z': qc.z}
for i in range(6):
    qc.h(i)

@app.route('/apply_gate', methods = ['POST'])
def apply_gate():
    data = request.get_json()
    gate = data['gate']
    qubit = data['qubit']
    gate_map[gate](qubit)
    return jsonify({'status': 'ok'})

@app.route('/generate', methods = ['GET'])
def generate():
    print("generate route hit")
    counts = measure_qc_1024()
    melody_sequence = []
    accompaniment_sequence = []
    for bitstring, count in counts.items():
        melody_bits = bitstring[:3]
        accompaniment_bits = bitstring[3:]
        melody_sequence.extend([melody_map[melody_bits]] * count)
        accompaniment_sequence.extend([accompaniment_map[accompaniment_bits]] * count)
    print(f"Melody:", melody_sequence)
    print("Accompaniment:", accompaniment_sequence)
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
 
if __name__ == '__main__':
    print('\n Qusic server starting...')
    print(' Open http://localhost:5000 in your browser.\n')
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)


