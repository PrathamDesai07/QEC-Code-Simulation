# === FILE: noise/bit_flip.py ===
import pennylane as qml

def apply_bit_flip(wires, p):
    for w in wires:
        qml.BitFlip(p, wires=w)