# === FILE: simulation/run_experiment.py ===
from codes.shor import shor_encoder, shor_decoder
from codes.steane import steane_encoder, steane_decoder
from codes.surface_code import surface_encoder, surface_decoder
from noise.bit_flip import apply_bit_flip
from noise.phase_flip import apply_phase_flip
from noise.depolarizing import apply_depolarizing
import numpy as np
import datetime
import pandas as pd
from functools import reduce

CSV_PATH = "results/simulations.csv"
LOG_PATH = "results/logs.csv"

def log_and_print(message):
    print(message)
    with open(LOG_PATH, "a") as f:
        f.write(message + "\n")

def is_valid_quantum_state(state):
    if state.ndim == 1:
        return np.allclose(np.linalg.norm(state), 1.0)
    elif state.ndim == 2:
        return np.allclose(np.trace(state), 1.0)
    return False

def fidelity_with_expected(rho, logical_bit):
    from pennylane import math
    expected = np.zeros_like(rho)
    expected[logical_bit, logical_bit] = 1.0
    return float(math.fidelity(rho, expected))

def build_operator(wires_list, num_wires):
    mats = []
    for i in range(num_wires):
        mats.append(np.array([[1, 0], [0, -1]]) if i in wires_list else np.eye(2))
    return reduce(lambda a, b: np.kron(a, b), mats)

def compute_syndrome(rho, stabilizers, num_wires):
    syndrome = []
    for wires in stabilizers:
        op = build_operator(wires, num_wires)
        val = np.real(np.trace(rho @ op))
        syndrome.append(val)
    return syndrome

def test_code(code_name, encoder, decoder, logical_bit=0, noise_type=None, p=0.0,
              stabilizers=None, num_wires=None, repeat=10):

    noise_fn = {
        "bit_flip": apply_bit_flip,
        "phase_flip": apply_phase_flip,
        "depolarizing": apply_depolarizing
    }.get(noise_type, None)

    results = []

    for trial in range(repeat):
        circuit = encoder()
        rho, _ = circuit(logical_bit=logical_bit, noise_fn=noise_fn, noise_args={"p": p})

        if not is_valid_quantum_state(rho):
            log_and_print(f"[ERROR] {code_name} - invalid state on trial {trial}")
            continue

        fid = fidelity_with_expected(rho, logical_bit)
        synd_vals = compute_syndrome(rho, stabilizers, num_wires)
        synd_bits = [1 if val < 0 else 0 for val in synd_vals]
        decoded = decoder(synd_bits)

        logical_error = fid < 0.9

        results.append({
            "code": code_name,
            "logical_bit": logical_bit,
            "noise_type": noise_type if noise_type else "none",
            "noise_prob": p,
            "syndrome": synd_bits,
            "decoded_result": decoded,
            "logical_error": logical_error,
            "fidelity": fid,
            "trial": trial
        })

    return results

# Stabilizers for each code
SHOR_STABILIZERS = [(0,1), (1,2), (3,4), (4,5), (6,7), (7,8)]
STEANE_STABILIZERS = [(0,1,2,3), (0,2,4,6), (1,2,5,6)]
SURFACE_STABILIZERS = [(0,1), (1,3)]

def main():
    with open(LOG_PATH, "a") as f:
        f.write(f"\n==== LOG START: {datetime.datetime.now()} ====\n")

    df_logs = []

    noise_models = [None, "bit_flip", "phase_flip", "depolarizing"]
    for noise in noise_models:
        for bit in [0, 1]:
            df_logs += test_code("Shor", shor_encoder, shor_decoder, bit, noise, p=0.9,
                                 stabilizers=SHOR_STABILIZERS, num_wires=9)
            df_logs += test_code("Steane", steane_encoder, steane_decoder, bit, noise, p=0.9,
                                 stabilizers=STEANE_STABILIZERS, num_wires=7)
            df_logs += test_code("Surface", surface_encoder, surface_decoder, bit, noise, p=0.9,
                                 stabilizers=SURFACE_STABILIZERS, num_wires=5)

    df = pd.DataFrame(df_logs)
    df.to_csv(CSV_PATH, index=False)
    log_and_print(f"Simulation results saved to: {CSV_PATH}")

if __name__ == "__main__":
    main()
