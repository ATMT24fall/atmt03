# Assignment03 - improving a low-resource NMT system
In the root directory locates several shell scripts helps to finish data preprocessing(BPE and BPE-drop-out), training, translation, postprocessing, evaluation. We finished two sets of experiments to improve the training and translation quality.
Following steps illustrate how to complete the experiments.
## Prerequisites
Make sure to run
```bash
pip install -r requirements.txt
```
to install all packages first.

## Steps to finish BPE preprocessing

### Change variables `BPE_OPERATIONS`, `VOCAB_THRESHOLD` in `bpe.sh` or `BPE_OPERATIONS`, `VOCAB_THRESHOLD`, `DROP_OUT` in `bpe_drop_out` and run 
```bash
bash bpe.sh
```
or
```bash
bash bpe_drop_out.sh
```
this will generate data folder with custom name in 'data/en-fr'.(e.g., 'data/en-fr/prepared_bpe_32000_15_0.1').

### Train on tiny or complete datasets by running
```bash
bash train_tiny_bpe.sh
``` 
or 
```bash
bash train_complete_bpe.sh
```
make sure to change variable `DATA_DIR` to your input data.


## Steps to train the model with different learning rate

### change variables `lr` in `train_tiny_lr.sh` or `train_complete_lr.sh`, adn run 
```bash
bash train_tiny_lr.sh
```
or 

```bash
bash train_complete_lr.sh
```

## Results
All checkpoints, translations, BLEU results will be stored in '/assignment03/baseline'