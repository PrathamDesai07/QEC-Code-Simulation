# === FILE: codes/steane.py ===
import pennylane as qml
from pennylane import numpy as np

def steane_encoder():
    dev = qml.device("default.mixed", wires=7)

    @qml.qnode(dev)
    def state_circuit(logical_bit=0, noise_fn=None, noise_args=None):
        if logical_bit == 1:
            qml.X(wires=0)

        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[0, 2])
        qml.CNOT(wires=[1, 3])
        qml.CNOT(wires=[1, 4])
        qml.CNOT(wires=[2, 5])
        qml.CNOT(wires=[2, 6])

        if noise_fn:
            noise_fn(wires=range(7), **(noise_args or {}))

        return qml.density_matrix(wires=range(7))

    @qml.qnode(dev)
    def syndrome_circuit():
        return [
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3)),
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(2) @ qml.PauliZ(4) @ qml.PauliZ(6)),
            qml.expval(qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(5) @ qml.PauliZ(6)),
        ]

    def combined_circuit(logical_bit=0, noise_fn=None, noise_args=None):
        state = state_circuit(logical_bit=logical_bit, noise_fn=noise_fn, noise_args=noise_args)
        syndrome = syndrome_circuit()
        return state, syndrome

    return combined_circuit

def steane_decoder(state):
    return state
