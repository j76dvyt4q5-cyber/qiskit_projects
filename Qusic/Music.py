from midiutil import MIDIFile
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile
import os

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
for i in range(6):
    qc.h(i)

if __name__ == '__main__':
    print('\n Qusic server starting...')
    print(' Open http://localhost:5000 in your browser.\n')
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
