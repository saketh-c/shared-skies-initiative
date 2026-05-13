#!/bin/bash
# Render all paper figures. Assumes results.json exists (run 00_fast_results.py first).
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f results.json ]; then
  echo "results.json not found. Run 00_fast_results.py first." >&2
  exit 1
fi
echo "Rendering figures..."
for script in 01_study_area.py 02_model_comparison.py 03_feature_importance.py \
              04_obs_vs_pred.py 05_ej_analysis.py 06_fairness_error.py \
              07_spatial_deployment.py; do
  echo "  $script ..."
  python3 "$script"
done
echo "✓ All figures rendered."
ls -la fig*.pdf
