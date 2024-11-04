## Steps to finish bpe preprocessing

### change variables in `bpe.sh` or `bpe_drop_out` and run 
```bash
bash bpe.sh
```
or
```bash
bash bpe_drop_out.sh
```

in terminal, this will generate data folder with custom name in 'data/en-fr'.(e.g., 'data/en-fr/prepared_bpe_32000_15_0.1')

### train on tiny or complete by running
```bash
bash train_tiny_bpe.sh
``` 
or 
```bash
bash train_complete_bpe.sh
```

## Steps to train the model with different learning rate

### change variables `lr` in `train_tiny_lr.sh` or `train_complete_lr.sh`, adn run 
```bash
bash train_tiny_lr.sh
```
or 

```bash
bash train_complete_lr.sh
```