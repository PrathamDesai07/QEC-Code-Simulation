# === FILE: codes/surface_code.py ===
import pennylane as qml
from pennylane import numpy as np

def surface_encoder():
    dev = qml.device("default.mixed", wires=5)

    @qml.qnode(dev)
    def state_circuit(logical_bit=0, noise_fn=None, noise_args=None):
        if logical_bit == 1:
            qml.X(wires=0)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[0, 2])
        qml.CNOT(wires=[1, 3])
        qml.CNOT(wires=[2, 4])

        if noise_fn:
            noise_fn(wires=range(5), **(noise_args or {}))

        return qml.density_matrix(wires=range(5))

    @qml.qnode(dev)
    def syndrome_circuit():
        return [
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
            qml.expval(qml.PauliZ(1) @ qml.PauliZ(3)),
        ]

    def combined_circuit(logical_bit=0, noise_fn=None, noise_args=None):
        state = state_circuit(logical_bit=logical_bit, noise_fn=noise_fn, noise_args=noise_args)
        syndrome = syndrome_circuit()
        return state, syndrome

    return combined_circuit

def surface_decoder(syndrome_bits):
    if syndrome_bits == [0, 0]:
        return "No error"
    elif syndrome_bits == [1, 0]:
        return "Possible X error on qubit 0 or 1"
    elif syndrome_bits == [0, 1]:
        return "Possible X error on qubit 1 or 3"
    elif syndrome_bits == [1, 1]:
        return "Multiple X errors or propagation"
    return "Unknown error pattern"