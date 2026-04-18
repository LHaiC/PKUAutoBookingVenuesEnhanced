#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/GLM-OCR}"
PORT="${PORT:-8000}"

python ocr_server_transformers.py --model "$MODEL_DIR" --port "$PORT"
