# === FILE: codes/shor.py ===
import pennylane as qml
from pennylane import numpy as np

def shor_encoder():
    dev = qml.device("default.mixed", wires=9)

    @qml.qnode(dev)
    def state_circuit(logical_bit=0, noise_fn=None, noise_args=None):
        if logical_bit == 1:
            qml.X(wires=0)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[0, 2])

        for i in [0, 1, 2]:
            qml.Hadamard(wires=i)
            qml.CNOT(wires=[i, i+3])
            qml.CNOT(wires=[i, i+6])
            qml.Hadamard(wires=i)

        if noise_fn:
            noise_fn(wires=range(9), **(noise_args or {}))

        return qml.density_matrix(wires=range(9))

    @qml.qnode(dev)
    def syndrome_circuit():
        return [
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
            qml.expval(qml.PauliZ(1) @ qml.PauliZ(2)),
            qml.expval(qml.PauliZ(3) @ qml.PauliZ(4)),
            qml.expval(qml.PauliZ(4) @ qml.PauliZ(5)),
            qml.expval(qml.PauliZ(6) @ qml.PauliZ(7)),
            qml.expval(qml.PauliZ(7) @ qml.PauliZ(8)),
        ]

    def combined_circuit(logical_bit=0, noise_fn=None, noise_args=None):
        state = state_circuit(logical_bit=logical_bit, noise_fn=noise_fn, noise_args=noise_args)
        syndrome = syndrome_circuit()
        return state, syndrome

    return combined_circuit

def shor_decoder(syndrome_bits):
    table = {
        (0, 0, 0, 0, 0, 0): "No error",
        (1, 0, 0, 0, 0, 0): "Bit-flip on qubit 0 or 1",
        (0, 1, 0, 0, 0, 0): "Bit-flip on qubit 1 or 2",
        (0, 0, 1, 0, 0, 0): "Bit-flip on qubit 3 or 4",
        (0, 0, 0, 1, 0, 0): "Bit-flip on qubit 4 or 5",
        (0, 0, 0, 0, 1, 0): "Bit-flip on qubit 6 or 7",
        (0, 0, 0, 0, 0, 1): "Bit-flip on qubit 7 or 8",
    }
    return table.get(tuple(syndrome_bits), "Unknown error pattern")
