#!/bin/bash
set -e

# ‼️ change this
DATA_DIR="data/en-fr/prepared_bpe_32000_15_0_1"
# Get BPE suffix from DATA_DIR
BPE_SUFFIX=$(echo $DATA_DIR | grep -o 'bpe_[0-9_]*')

# Directory setup
BASE_DIR="assignment03/${BPE_SUFFIX}_tiny"
CHECKPOINTS_DIR="$BASE_DIR/checkpoints"
TRANSLATIONS_FILE="$BASE_DIR/translations.txt"
POSTPROCESSED_FILE="$BASE_DIR/translations.p.txt"

# Create necessary directories
mkdir -p "$CHECKPOINTS_DIR"

echo "Starting tiny training pipeline..."

# Train the model on tiny dataset
echo "Training model on tiny dataset..."
python train.py \
    --data "$DATA_DIR" \
    --source-lang fr \
    --target-lang en \
    --save-dir "$CHECKPOINTS_DIR" \
    --train-on-tiny

# Translate using trained model
echo "Translating test set..."
python translate.py \
    --data "$DATA_DIR" \
    --dicts "$DATA_DIR" \
    --checkpoint-path "$CHECKPOINTS_DIR/checkpoint_last.pt" \
    --output "$TRANSLATIONS_FILE"

# Postprocess translations
echo "Postprocessing translations..."
bash scripts/postprocess.sh \
    "$TRANSLATIONS_FILE" \
    "$POSTPROCESSED_FILE" \
    en

# Evaluate using BLEU score and save results
echo "Evaluating translations..."
RESULTS_FILE="$BASE_DIR/bleu_results.txt"
echo "BLEU Score Evaluation Results - $(date)" > "$RESULTS_FILE"
cat "$POSTPROCESSED_FILE" | sacrebleu data/en-fr/raw/test.en | tee -a "$RESULTS_FILE"

echo "Tiny training pipeline completed!" 