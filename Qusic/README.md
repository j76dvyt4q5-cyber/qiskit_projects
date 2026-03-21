---
title: Qusic
emoji: 🎵
colorFrom: green
colorTo: gray
sdk: docker
pinned: false
---
# Qusic — Quantum Music Generator

A music generator where every note is determined by collapsing a real quantum circuit using Qiskit and AerSimulator. Apply H, X, Z, RY, and RZ gates to shape the probability distribution of pitches across a 10-qubit circuit — then measure to generate a two-voice MIDI file.

- Qubits 0–4 control the melody register
- Qubits 5–9 control the accompaniment register
- Real-time probability display updates with every gate applied
- GHZ state entangles both voices for maximum correlation
- Export MIDI and open in Logic Pro, GarageBand, or any DAW
