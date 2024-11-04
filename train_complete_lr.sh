#!/bin/bash

# ‼️ change this
# Set the learning rate
lr=0.0005
echo "Learning rate set to $lr"

baseline_dir="assignment03/baseline/lr_${lr}/"
echo "Baseline directory: $baseline_dir"


checkpoint_dir="${baseline_dir}checkpoints/"
echo "Checkpoint directory: $checkpoint_dir"

# Create the checkpoint directory if it doesn't exist
echo "Creating directories if they don't exist..."
mkdir -p $baseline_dir
mkdir -p $checkpoint_dir

# Train the model
echo "Starting model training..."
python train.py \
    --data data/en-fr/prepared \
    --source-lang fr \
    --target-lang en \
    --save-dir $checkpoint_dir \
    --lr $lr

# Translate using the last checkpoint
echo "Translating using the last checkpoint..."
python translate.py \
    --data data/en-fr/prepared/ \
    --dicts data/en-fr/prepared/ \
    --checkpoint-path $checkpoint_dir/checkpoint_last.pt \
    --output $baseline_dir/translations.txt

# Post-process the translations
echo "Post-processing translations..."
bash scripts/postprocess.sh \
    $baseline_dir/translations.txt \
    $baseline_dir/translations.p.txt en

# Evaluate using BLEU score and save results
echo "Evaluating translations..."
RESULTS_FILE="$baseline_dir/bleu_results.txt"
echo "BLEU Score Evaluation Results - $(date)"
cat $baseline_dir/translations.p.txt | sacrebleu data/en-fr/raw/test.en | tee -a "$RESULTS_FILE"

echo "Tiny training pipeline completed!" 