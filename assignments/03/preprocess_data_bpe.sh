#!/bin/bash
# -*- coding: utf-8 -*-

set -e
set -o pipefail
set -u

# Get script directory and base paths
pwd="$(dirname "$(readlink -f "$0")")"
base="$pwd/../.."
src=fr
tgt=en
data="$base/data/en-fr"
moses_scripts="$base/moses_scripts"
preprocessed="$data/preprocessed_bpe"
prepared="$data/prepared_bpe"

# Add new variables for BPE
BPE_OPS=32000           # Reduced from 50000 for better subword units
BPE_MIN_FREQ=5         # Minimum frequency for BPE merges
TRAIN_DROPOUT=0.1      # Reduced from 0.3 for more stable training
VOCAB_SIZE=32000       # Vocabulary size limit
MIN_SEQ_LENGTH=1       # Remove sequences shorter than this
MAX_SEQ_LENGTH=250     # Remove sequences longer than this

# Function to check if required tools exist
check_requirements() {
    local required_tools=("perl" "subword-nmt" "python" "awk")
    local required_scripts=(
        "$moses_scripts/normalize-punctuation.perl"
        "$moses_scripts/tokenizer.perl"
        "$moses_scripts/train-truecaser.perl"
        "$moses_scripts/truecase.perl"
    )

    echo "Checking requirements..."
    # Check tools
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            echo "Error: Required tool '$tool' is not installed"
            exit 1
        fi
    done

    # Check Moses scripts
    for script in "${required_scripts[@]}"; do
        if [[ ! -f "$script" ]]; then
            echo "Error: Required Moses script not found: $script"
            exit 1
        fi
    done
}

# Function to create required directories
setup_directories() {
    local dirs=("$preprocessed" "$prepared" "$data/raw")
    echo "Setting up directories..."
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
    done
}

# Function to check if input files exist
check_input_files() {
    local splits=("train" "valid" "test" "tiny_train")
    echo "Checking input files..."
    for split in "${splits[@]}"; do
        for lang in "$src" "$tgt"; do
            if [[ ! -f "$data/raw/$split.$lang" ]]; then
                echo "Error: Input file not found: $data/raw/$split.$lang"
                exit 1
            fi
        done
    done
}

# Function to preprocess one file
preprocess_file() {
    local input=$1
    local output=$2
    local lang=$3
    local truecaser_model=$4

    echo "Preprocessing $input to $output"
    perl "$moses_scripts/normalize-punctuation.perl" -l "$lang" < "$input" | \
    perl "$moses_scripts/tokenizer.perl" -l "$lang" -a -q | \
    perl "$moses_scripts/truecase.perl" --model "$truecaser_model" > "$output"
}

# Function for joint BPE learning
learn_joint_bpe() {
    local src_file=$1
    local tgt_file=$2
    local codes_file=$3
    local vocab_size=$4

    echo "Learning joint BPE with vocabulary size $vocab_size..."
    subword-nmt learn-joint-bpe-and-vocab \
        --input $src_file $tgt_file \
        -s $vocab_size \
        -o $codes_file \
        --write-vocabulary $preprocessed/vocab.$src $preprocessed/vocab.$tgt \
        --min-frequency $BPE_MIN_FREQ
        --total-symbols  # Add this flag
        --num-workers 8  # Add parallel processing
}

# Function to apply BPE with progress
apply_bpe_with_progress() {
    local input=$1
    local output=$2
    local codes=$3
    local vocab=$4
    local dropout=$5
    local total_lines=$(wc -l < $input)
    local current_line=0

    echo "Applying BPE to $(basename $input)..."
    
    # Apply BPE with vocabulary filtering
    if [[ -n "$dropout" ]]; then
        subword-nmt apply-bpe \
            -c $codes \
            --vocabulary $vocab \
            --vocabulary-threshold 10 \
            --dropout $dropout \
            < $input > $output
    else
        subword-nmt apply-bpe \
            -c $codes \
            --vocabulary $vocab \
            --vocabulary-threshold 10 \
            < $input > $output
    fi
}

# Function to clean and validate sequences
clean_sequences() {
    local input=$1
    local output=$2
    
    echo "Cleaning sequences in $(basename $input)..."
    awk -v min=$MIN_SEQ_LENGTH -v max=$MAX_SEQ_LENGTH '
    {
        len = NF
        if (len >= min && len <= max) {
            print
        }
    }' $input > $output
}

# Function to validate BPE application
validate_bpe() {
    local file=$1
    local split=$2
    local lang=$3
    
    echo "Validating BPE application in $split.$lang..."
    python -c "
import sys
with open('$file', 'r') as f:
    for i, line in enumerate(f, 1):
        if '@@' not in line and i < 100:
            print(f'Warning: No BPE splits found in first {i} lines of $split.$lang')
            break
"
}

# Main preprocessing function
main() {
    # Initial checks
    check_requirements
    setup_directories
    check_input_files

    cd "$base"
    echo "Starting preprocessing pipeline..."

    # Step 1: Normalize and tokenize training data
    echo "Normalizing and tokenizing training data..."
    for lang in "$src" "$tgt"; do
        cat "$data/raw/train.$lang" | \
        perl "$moses_scripts/normalize-punctuation.perl" -l "$lang" | \
        perl "$moses_scripts/tokenizer.perl" -l "$lang" -a -q > "$preprocessed/train.$lang.p"
    done

    # Step 2: Train truecaser models
    echo "Training truecaser models..."
    for lang in "$src" "$tgt"; do
        perl "$moses_scripts/train-truecaser.perl" \
            --model "$preprocessed/tm.$lang" \
            --corpus "$preprocessed/train.$lang.p"
    done

    # Step 3: Apply truecaser to training data
    echo "Applying truecaser to training data..."
    for lang in "$src" "$tgt"; do
        perl "$moses_scripts/truecase.perl" \
            --model "$preprocessed/tm.$lang" \
            < "$preprocessed/train.$lang.p" \
            > "$preprocessed/train.$lang"
    done

    # Step 4: Process validation and test sets
    echo "Processing validation and test sets..."
    for split in valid test tiny_train; do
        for lang in "$src" "$tgt"; do
            preprocess_file \
                "$data/raw/$split.$lang" \
                "$preprocessed/$split.$lang" \
                "$lang" \
                "$preprocessed/tm.$lang"
        done
    done

    # Step 5: Learn joint BPE for both languages
    echo "Learning joint BPE model..."
    learn_joint_bpe \
        "$preprocessed/train.$src" \
        "$preprocessed/train.$tgt" \
        "$preprocessed/bpe.codes" \
        $BPE_OPS

    # Step 6: Apply BPE to all splits with improved handling
    echo "Applying BPE to all splits..."
    for split in train valid test tiny_train; do
        for lang in "$src" "$tgt"; do
            # Clean sequences before BPE
            clean_sequences \
                "$preprocessed/$split.$lang" \
                "$preprocessed/$split.cleaned.$lang"

            if [[ "$split" == "train" ]]; then
                # Apply BPE with dropout to training data
                apply_bpe_with_progress \
                    "$preprocessed/$split.cleaned.$lang" \
                    "$preprocessed/$split.bpe.$lang" \
                    "$preprocessed/bpe.codes" \
                    "$preprocessed/vocab.$lang" \
                    $TRAIN_DROPOUT
            else
                # Apply BPE without dropout to other splits
                apply_bpe_with_progress \
                    "$preprocessed/$split.cleaned.$lang" \
                    "$preprocessed/$split.bpe.$lang" \
                    "$preprocessed/bpe.codes" \
                    "$preprocessed/vocab.$lang" \
                    ""
            fi
            
            # Validate BPE application
            validate_bpe \
                "$preprocessed/$split.bpe.$lang" \
                "$split" \
                "$lang"
        done
    done

    # Step 7: Clean up temporary files
    echo "Cleaning up temporary files..."
    rm -f "$preprocessed/train.$src.p" "$preprocessed/train.$tgt.p"
    rm -f "$preprocessed/"*.cleaned.*

    # Step 8: Final preprocessing with improved vocabulary handling
    echo "Running final preprocessing for model training..."
    python preprocess.py \
        --target-lang "$tgt" \
        --source-lang "$src" \
        --dest-dir "$prepared" \
        --train-prefix "$preprocessed/train.bpe" \
        --valid-prefix "$preprocessed/valid.bpe" \
        --test-prefix "$preprocessed/test.bpe" \
        --tiny-train-prefix "$preprocessed/tiny_train.bpe" \
        --threshold-src 2 \
        --threshold-tgt 2 \
        --num-words-src $VOCAB_SIZE \
        --num-words-tgt $VOCAB_SIZE

    echo "BPE Preprocessing completed successfully!"
}

# Run main function with error handling
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    trap 'echo "Error on line $LINENO. Exit code: $?"' ERR
    main "$@"
fi