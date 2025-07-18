
# === FILE: codes/surface_code.py ===
import pennylane as qml
from pennylane import numpy as np
import pymatching

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
    rounded = [int(bit > 0.5) for bit in syndrome_bits]

    H = [
        [1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
    ]

    matching = pymatching.Matching(H)
    correction = matching.decode(rounded)

    if not any(correction):
        return "No error"

    qubits = [i for i, bit in enumerate(correction) if bit]
    return f"MWPM corrected qubit(s): {qubits}"
