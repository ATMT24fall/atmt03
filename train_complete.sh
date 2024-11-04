#!/bin/bash
set -e

# Directory setup
BASE_DIR="assignment03/baseline"
CHECKPOINTS_DIR="$BASE_DIR/checkpoints"
DATA_DIR="data/en-fr/prepared_bpe"
TRANSLATIONS_FILE="$BASE_DIR/translations.txt"
POSTPROCESSED_FILE="$BASE_DIR/translations.p.txt"

# Create necessary directories
mkdir -p "$CHECKPOINTS_DIR"

echo "Starting complete training pipeline..."

# Check and remove incompatible checkpoints
if [ -f "$CHECKPOINTS_DIR/checkpoint_last.pt" ]; then
    echo "Removing incompatible checkpoint..."
    rm "$CHECKPOINTS_DIR/checkpoint_last.pt"
fi

# Check and remove incompatible translations
if [ -f "$TRANSLATIONS_FILE" ]; then
    echo "Removing incompatible translations..."
    rm "$TRANSLATIONS_FILE"
fi

# Train the model on complete dataset
echo "Training model on complete dataset..."
python train.py \
    --data "$DATA_DIR" \
    --source-lang fr \
    --target-lang en \
    --save-dir "$CHECKPOINTS_DIR"

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

# Evaluate using BLEU score
echo "Evaluating translations..."
cat "$POSTPROCESSED_FILE" | sacrebleu data/en-fr/raw/test.en

echo "Complete training pipeline completed!" 