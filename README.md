# Quantum Error Correction Project (Shor, Steane, Surface Codes)

This project simulates and compares Shor, Steane, and Surface quantum error correction codes using PennyLane.

## Structure
- `codes/`: Encoding circuits
- `noise/`: Noise models (bit-flip, phase-flip, depolarizing)
- `decoders/`: Decoders (lookup tables, surface MWPM)
- `simulation/`: Run experiments and plot results
- `results/`: Collected data and plots
- `utils/`: Helper functions

## Requirements
Install all dependencies:
```bash
pip install -r requirements.txt
```

## Run
To simulate all codes:
```bash
python simulation/run_experiment.py
```
