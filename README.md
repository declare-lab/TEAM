# Two is Better than Many? Binary Classification as an Effective Approach to Multi-Choice Question Answering

The repository contains the implementation of the paper [Two is Better than Many? Binary Classification as an Effective Approach to Multi-Choice Question Answering](https://arxiv.org/abs/2210.16495) (EMNLP 2022)

## Experiments

We have created separate training scripts for each of the datasets.

For example, the DeBERTa TEAM model on the SWAG dataset can be trained as follows:

```
CUDA_VISIBLE_DEVICES=0 python train_swag.py --name "microsoft/deberta-v3-large" --epochs 5 --lr 1e-6 --shuffle
```

You can use `--name "roberta-large"` to train the RoBERTa model.

You can use the appropriate training scripts for the other datasets.

## Citation

Please cite the following paper if you find this code useful in your work.

```
Two is Better than Many? Binary Classification as an Effective Approach to Multi-Choice Question Answering. D. Ghosal, N. Majumder, R. Mihalcea, S. Poria. EMNLP 2022.
```