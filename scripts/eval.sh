#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/train_mimic_pa_512_ema.yaml}

python -m src.evaluate --config "${CONFIG}"
