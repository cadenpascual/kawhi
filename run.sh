#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting K.A.W.H.I. Pipeline..."

# 1. Download and parse raw data
echo "Extracting 7z files..."
python src/scripts/7z_to_json.py

# 2. Build defensive features and IST variables
echo "Generating defensive features..."
python src/pipelines/defense_parquet.py
# python src/pipelines/calculate_ist.py  <-- (If you convert notebook 06/07 to a script)

# 3. Run the Wasserstein Gradient Flow Optimization
echo "Running optimization simulations..."
python src/gradient_flows/optimize.py

echo "Pipeline complete! Results saved to data/processed/optimization/"