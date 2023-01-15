# Two is Better than Many? Binary Classification as an Effective Approach to Multi-Choice Question Answering

The repository contains the implementation of the paper [Two is Better than Many? Binary Classification as an Effective Approach to Multi-Choice Question Answering](https://arxiv.org/abs/2210.16495) (EMNLP 2022)

## Experiments

We have created separate training scripts for each of the datasets.

For example, the DeBERTa TEAM model on the SWAG dataset can be trained as follows:

```
CUDA_VISIBLE_DEVICES=0 python train_swag.py --name "microsoft/deberta-v3-large" --epochs 5 --lr 1e-6 --shuffle
```

You can use `--name "roberta-large"` to train the RoBERTa model.

You can use the appropriate training scripts for the other datasets. Running the scripts will print an `Instance Acc`, which is the main MCQA task accuracy reported in the Table 2 and 3 of our paper. For some of the datasets, you need to upload the test predictions in the [AllenAI Leaderboard](https://leaderboard.allenai.org/) to obtain the test results. The scripts provided in this repository will save the test predictions for each epoch in the appropriate experiment folders ready for upload to the leaderboard.


The Score models can be benchmarked using the `run_mcqa_score.py` script. The scirpt is adapted from the [HuggingFace MCQA example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice).

The DeBERTa Score model on the SWAG dataset can be trained as follows:
```
CUDA_VISIBLE_DEVICES=3 python run_mcqa_score.py --learning_rate=1e-6 --num_train_epochs 5 --seed 42 \
--train_file="data/swag/mcq_train.json" --validation_file="data/swag/mcq_val.json" --test_file="data/swag/mcq_test.json" \
--output_dir="saved/swag/mcq/deberta-large" --model_name_or_path="microsoft/deberta-v3-large" \
--per_device_train_batch_size=8 --per_device_eval_batch_size=8 --weight_decay=0.005 \
--do_train True --do_eval True --do_predict True --evaluation_strategy="epoch" --save_strategy="epoch" \
--report_to "wandb" --run_name "DEBERTA SWAG MCQ" --save_total_limit=1 --overwrite_output_dir
```

Change the `--train_file, --validation_file, --test_file` arguments to train and evaluate on the other datasets. Change the `--model_name_or_path` to train other models for the task.

## Citation

Please cite the following paper if you find this code useful in your work.

```
Two is Better than Many? Binary Classification as an Effective Approach to Multi-Choice Question Answering. D. Ghosal, N. Majumder, R. Mihalcea, S. Poria. EMNLP 2022.
```