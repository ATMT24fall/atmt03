# Assignment03 - Low-Resource NMT System Improvement

This repository contains a collection of shell scripts for improving a low-resource Neural Machine Translation (NMT) system. The scripts handle data preprocessing (including BPE and BPE-dropout), model training, translation, postprocessing, and evaluation. We conducted two sets of experiments to enhance training and translation quality.

## Prerequisites

Install all required packages by running:
```bash
pip install -r requirements.txt
```

## Data Preprocessing with BPE

### Standard BPE
1. Configure the following variables in `bpe.sh`:
   - `BPE_OPERATIONS`: Number of BPE merge operations
   - `VOCAB_THRESHOLD`: Vocabulary threshold

2. Run the preprocessing:
```bash
bash bpe.sh
```

### BPE with Dropout
1. Configure the following variables in `bpe_drop_out.sh`:
   - `BPE_OPERATIONS`: Number of BPE merge operations
   - `VOCAB_THRESHOLD`: Vocabulary threshold
   - `DROP_OUT`: Dropout rate

2. Run the preprocessing:
```bash
bash bpe_drop_out.sh
```

The preprocessed data will be generated in `/data/en-fr` with a custom name based on your parameters (e.g., `data/en-fr/prepared_bpe_32000_15_0.1`).

## Model Training

### Training with Different Dataset Sizes

1. For the tiny dataset:
```bash
bash train_tiny_bpe.sh
```

2. For the complete dataset:
```bash
bash train_complete_bpe.sh
```

**Note**: Make sure to update the `DATA_DIR` variable in the respective scripts to point to your preprocessed data directory.

### Training with Different Learning Rates

1. Modify the `lr` variable in either:
   - `train_tiny_lr.sh` (for tiny dataset)
   - `train_complete_lr.sh` (for complete dataset)

2. Run the training:
```bash
bash train_tiny_lr.sh
```
or
```bash
bash train_complete_lr.sh
```

## Output Directory Structure

All experimental outputs are stored in `/assignment03/{experiment_folder}/`, including:
- Model checkpoints
- Translation outputs
- BLEU score evaluations