## Steps to finish bpe preprocessing

### change variables in `bpe.sh` and run 
```bash
bash bpe.sh
```
in terminal, this will generate data folder with custom name in 'data/en-fr'.(e.g., 'data/en-fr/prepared_bpe_32000_15_0.1')

### train on tiny or complete by running
```bash
bash train_tiny.sh
``` 
or 
```bash
"bash train_complete.sh
```

## Steps to train the model with different learning rate

### change variables `lr` in `train_translate_evaluate_tiny.sh` or `train_translate_evaluate.sh`, adn run 
```bash
bash train_translate_evaluate_tiny.sh
```
or 

```bash
bash train_translate_evaluate.sh
```