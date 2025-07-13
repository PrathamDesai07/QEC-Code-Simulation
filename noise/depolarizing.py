
# === FILE: noise/depolarizing.py ===
import pennylane as qml

def apply_depolarizing(wires, p):
    for w in wires:
        qml.DepolarizingChannel(p, wires=w)