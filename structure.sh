#!/bin/bash

# Project root folder
PROJECT_NAME="."
cd "$PROJECT_NAME"

# Subfolders
mkdir -p "$PROJECT_NAME"/{codes,noise,decoders,simulation,results/{plots,tables},utils}

# Core Python files (with basic headers)
touch "$PROJECT_NAME"/codes/{shor.py,steane.py,surface_code.py}
touch "$PROJECT_NAME"/noise/{bit_flip.py,phase_flip.py,depolarizing.py}
touch "$PROJECT_NAME"/decoders/{lookup_decoder.py,surface_decoder.py}
touch "$PROJECT_NAME"/simulation/{run_experiment.py,plot_results.py}
touch "$PROJECT_NAME"/utils/helpers.py

# Results files
touch "$PROJECT_NAME"/results/logs.csv

# Documentation and requirements
touch "$PROJECT_NAME"/{README.md,requirements.txt}

# Add starter content to README
cat <<EOL > "$PROJECT_NAME"/README.md
# Quantum Error Correction Project (Shor, Steane, Surface Codes)

This project simulates and compares Shor, Steane, and Surface quantum error correction codes using PennyLane.

## Structure
- \`codes/\`: Encoding circuits
- \`noise/\`: Noise models (bit-flip, phase-flip, depolarizing)
- \`decoders/\`: Decoders (lookup tables, surface MWPM)
- \`simulation/\`: Run experiments and plot results
- \`results/\`: Collected data and plots
- \`utils/\`: Helper functions

## Requirements
Install all dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Run
To simulate all codes:
\`\`\`bash
python simulation/run_experiment.py
\`\`\`
EOL

# Add starter content to requirements.txt
cat <<EOL > "$PROJECT_NAME"/requirements.txt
pennylane==0.36.0
numpy>=1.24
matplotlib>=3.5
pandas>=1.5
seaborn>=0.11
pymatching==2.1.0
tqdm>=4.64
scipy>=1.10
EOL

echo "✅ Project structure for Quantum Error Correction created successfully in ./$PROJECT_NAME"
