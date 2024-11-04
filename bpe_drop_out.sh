# ‼️ change this
OUTPUT_DIR=./data/en-fr/prepared_bpe_32000_15_0_1

# Define BPE operations and vocabulary threshold variables

# ‼️ change this
BPE_OPERATIONS=32000
VOCAB_THRESHOLD=15
DROP_OUT=0.1
# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Learn BPE with adjusted parameters
subword-nmt learn-joint-bpe-and-vocab \
    --input ./data/en-fr/preprocessed/train.en ./data/en-fr/preprocessed/train.fr \
    -s $BPE_OPERATIONS \
    -o $OUTPUT_DIR/codes.txt \
    --write-vocabulary $OUTPUT_DIR/vocab.en $OUTPUT_DIR/vocab.fr

# Apply BPE to train files
subword-nmt apply-bpe \
    -c $OUTPUT_DIR/codes.txt \
    --dropout $DROP_OUT \
    --vocabulary $OUTPUT_DIR/vocab.en \
    --vocabulary-threshold $VOCAB_THRESHOLD \
    < ./data/en-fr/preprocessed/train.en \
    > $OUTPUT_DIR/train.en

subword-nmt apply-bpe \
    -c $OUTPUT_DIR/codes.txt \
    --dropout $DROP_OUT \
    --vocabulary $OUTPUT_DIR/vocab.fr \
    --vocabulary-threshold $VOCAB_THRESHOLD \
    < ./data/en-fr/preprocessed/train.fr \
    > $OUTPUT_DIR/train.fr

# Apply BPE to tiny_train
subword-nmt apply-bpe \
    -c $OUTPUT_DIR/codes.txt \
    --vocabulary $OUTPUT_DIR/vocab.en \
    --vocabulary-threshold $VOCAB_THRESHOLD \
    < ./data/en-fr/preprocessed/tiny_train.en \
    > $OUTPUT_DIR/tiny_train.en

subword-nmt apply-bpe \
    -c $OUTPUT_DIR/codes.txt \
    --vocabulary $OUTPUT_DIR/vocab.fr \
    --vocabulary-threshold $VOCAB_THRESHOLD \
    < ./data/en-fr/preprocessed/tiny_train.fr \
    > $OUTPUT_DIR/tiny_train.fr

# Apply BPE to valid
subword-nmt apply-bpe \
    -c $OUTPUT_DIR/codes.txt \
    --vocabulary $OUTPUT_DIR/vocab.en \
    --vocabulary-threshold $VOCAB_THRESHOLD \
    < ./data/en-fr/preprocessed/valid.en \
    > $OUTPUT_DIR/valid.en

subword-nmt apply-bpe \
    -c $OUTPUT_DIR/codes.txt \
    --vocabulary $OUTPUT_DIR/vocab.fr \
    --vocabulary-threshold $VOCAB_THRESHOLD \
    < ./data/en-fr/preprocessed/valid.fr \
    > $OUTPUT_DIR/valid.fr

# Apply BPE to test files
subword-nmt apply-bpe \
    -c $OUTPUT_DIR/codes.txt \
    --vocabulary $OUTPUT_DIR/vocab.en \
    --vocabulary-threshold $VOCAB_THRESHOLD \
    < ./data/en-fr/preprocessed/test.en \
    > $OUTPUT_DIR/test.en

subword-nmt apply-bpe \
    -c $OUTPUT_DIR/codes.txt \
    --vocabulary $OUTPUT_DIR/vocab.fr \
    --vocabulary-threshold $VOCAB_THRESHOLD \
    < ./data/en-fr/preprocessed/test.fr \
    > $OUTPUT_DIR/test.fr

# Generate dictionaries for NMT using the BPE-processed data
python preprocess.py \
    --source-lang fr \
    --target-lang en \
    --train-prefix $OUTPUT_DIR/train \
    --tiny-train-prefix $OUTPUT_DIR/tiny_train \
    --valid-prefix $OUTPUT_DIR/valid \
    --test-prefix $OUTPUT_DIR/test \
    --dest-dir $OUTPUT_DIR \
    --threshold-src 5 \
    --threshold-tgt 5 \
    --num-words-src 24000 \
    --num-words-tgt 24000
