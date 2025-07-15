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

import pymatching

def surface_decoder(syndrome_bits):
    # Hardcoded 2D surface code graph with 2 stabilizers and 3 edges (simplified)
    # This is only for a minimal toy model. For real codes, this graph must be built properly.
    
    # Syndrome indices match stabilizers [Z0Z1, Z1Z3]
    # Qubits: 0, 1, 2, 3, 4 (data)
    # Stabilizers: s0 = Z0Z1, s1 = Z1Z3

    # Define parity check matrix (rows: stabilizers, cols: qubits)
    H = [
        [1, 1, 0, 0, 0],  # Z0Z1: touches qubits 0,1
        [0, 1, 0, 1, 0],  # Z1Z3: touches qubits 1,3
    ]

    matching = pymatching.Matching(H)
    
    # MWPM decoding: input is syndrome bits
    correction = matching.decode(syndrome_bits)

    if not any(correction):
        return "No error"

    # Identify which qubits were corrected
    qubits = [i for i, bit in enumerate(correction) if bit]
    return f"MWPM corrected qubit(s): {qubits}"
