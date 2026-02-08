#!/usr/bin/env bash
set -euo pipefail

MIMIC_ROOT=${1:-/scratch/pkrish52/MIMIC}
OUT_DIR=${2:-./data/processed_pa}

python -m src.prepare_mimic       --mimic_root "${MIMIC_ROOT}"       --output_dir "${OUT_DIR}"       --views PA       --uncertainty_policy u_zeros
