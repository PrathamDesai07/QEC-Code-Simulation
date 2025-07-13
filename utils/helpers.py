# === FILE: utils/helpers.py ===
def compare_states(state1, state2, atol=1e-2):
    from numpy import allclose
    return allclose(state1, state2, atol=atol)
