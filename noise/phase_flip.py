# === FILE: noise/phase_flip.py ===
import pennylane as qml

def apply_phase_flip(wires, p):
    for w in wires:
        qml.PhaseFlip(p, wires=w)