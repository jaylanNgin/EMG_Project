#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$PWD}"

cd "$PROJECT_DIR"
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip libopenblas-dev

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lda.txt

echo
echo "Raspberry Pi LDA environment is ready."
echo "Activate it with: source .venv/bin/activate"
