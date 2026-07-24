#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

case "$MODE" in
  off|OFF|gloveoff|GloveOff|GLOVEOFF)
    SUBJECT_DIR="Subject2GloveOff"
    BEST_SIZE=200
    ;;
  on|ON|gloveon|GloveOn|GLOVEON)
    SUBJECT_DIR="Subject2GloveOn"
    BEST_SIZE=300
    ;;
  fast|FAST|Fast)
    SUBJECT_DIR="subjectFast"
    BEST_SIZE=300
    ;;
  norm|NORM|Norm|normal|NORMAL|Normal)
    SUBJECT_DIR="SubjectNorm"
    BEST_SIZE=200
    ;;
  slow|SLOW|Slow)
    SUBJECT_DIR="SubjectSlow"
    BEST_SIZE=250
    ;;
  *)
    echo "Usage: $0 <off|on|fast|norm|slow>"
    exit 2
    ;;
esac

DATASET="${SUBJECT_DIR}/ResultClipSizeUp${BEST_SIZE}"

echo "Running LDA for ${SUBJECT_DIR} with clip size ${BEST_SIZE}..."
MPLBACKEND="${MPLBACKEND:-Agg}" "$PYTHON_BIN" scripts/classify_emg_lda.py \
  --data-dir "$DATASET/act1" "$DATASET/act2" \
  --folder-labels
