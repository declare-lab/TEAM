#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for multiple choice.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

import logging
import os
import sys
import time
import json
from dataclasses import dataclass, field
from typing import Optional, Union
from datetime import datetime
from pathlib import Path

import pickle
import datasets
import numpy as np
import torch
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2ForMultipleChoice
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from sklearn.metrics import accuracy_score

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.13.0.dev0")

logger = logging.getLogger(__name__)

# os.environ["WANDB_DISABLED"] = "true"

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file for inference."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If passed, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to the maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    def __post_init__(self):
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.test_file is not None:
            extension = self.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.train_file is not None or data_args.validation_file is not None or data_args.test_file is not None:
        # print ("Training on ANLI")
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    else:
        # Downloading and loading the swag dataset from the hub.
        print ("Training on SWAG")
        raw_datasets = load_dataset("swag", "regular", cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    
    if "deberta" in model_args.model_name_or_path:
        mname = "microsoft/deberta-v3-large"
    elif "roberta" in model_args.model_name_or_path:
        mname = "roberta-large"
    
    
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else mname,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else mname,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if "deberta" in model_args.model_name_or_path:
        model = DebertaV2ForMultipleChoice.from_pretrained(
            # model_args.model_name_or_path,
            mname,
            from_tf=False,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForMultipleChoice.from_pretrained(
            # model_args.model_name_or_path,
            mname,
            from_tf=False,
            # from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        
    '''
    if "saved" in model_args.model_name_or_path:
        x = torch.load(model_args.model_name_or_path)
        pretrained_dict = {k[6:]: v for k, v in x.items() if ("classifier" not in k and "scorer" not in k)}
        
        model_dict = model.state_dict().copy()
        for k in model_dict:
            if "classifier" in k:
                pretrained_dict[k] = model_dict[k]
        model.load_state_dict(pretrained_dict)
    '''

    # When using your own dataset or a different dataset from swag, you will probably need to change this.
    
    if "csqa2" in data_args.train_file:
        n_choices = 2
    elif "csqa" in data_args.train_file:
        n_choices = 5
    elif "piqa" in data_args.train_file:
        n_choices = 2
    elif "siqa" in data_args.train_file:
        n_choices = 3
    elif "hellaswag" in data_args.train_file:
        n_choices = 4
    elif "swag" in data_args.train_file:
        n_choices = 4
    elif "cosmosqa" in data_args.train_file:
        n_choices = 4
    elif "qasc" in data_args.train_file:
        n_choices = 8
    elif "cicero2" in data_args.train_file:
        n_choices = 4
    elif "cicero" in data_args.train_file:
        n_choices = 5
    
        
    ending_names = [f"choice{i}" for i in range(n_choices)]
    context_name = "context"

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        else:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is shorter than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={data_args.max_seq_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Preprocessing the datasets.
    def preprocess_function(examples):
        # print (examples)
        first_sentences = [[context] * n_choices for context in examples[context_name]]
        second_sentences = [
            [f"{examples[end][i]}" for end in ending_names] for i in range(len(examples[context_name]))
        ]

        # Flatten out
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
        )
        # Un-flatten
        return {k: [v[i : i + n_choices] for i in range(0, len(v), n_choices)] for k, v in tokenized_examples.items()}

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                #num_proc=data_args.preprocessing_num_workers,
                #load_from_cache_file=not data_args.overwrite_cache,
            )

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                #num_proc=data_args.preprocessing_num_workers,
                #load_from_cache_file=not data_args.overwrite_cache,
            )

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = raw_datasets["test"]
        if data_args.max_eval_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="test dataset map pre-processing"):
            test_dataset = test_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    # Data collator
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorForMultipleChoice(tokenizer=tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )
    
    global epoch_count
    global running_set
    global run_id
    global dataset
    global path
    global fname
    global lf_name

    epoch_count = 0
    running_set = "validation"
    
    print ("Epoch count", epoch_count)
    
    name = model_args.model_name_or_path
    exp_id = str(int(time.time()))
    dataset = data_args.train_file.split("/")[1]
    arguments = {"name": name, "lr": training_args.learning_rate}

    if training_args.do_train:
        path = "saved/" + dataset + "/mcq/" + exp_id + "/" + name.replace("/", "-")
        Path("saved/" + dataset + "/mcq/" + exp_id + "/").mkdir(parents=True, exist_ok=True)
        Path("results/" + dataset + "/mcq/").mkdir(parents=True, exist_ok=True)
    
        fname = "saved/" + dataset + "/mcq/" + exp_id + "/" + "args.txt"
        f = open(fname, "a")
        f.write(str(arguments) + "\n\n")
        f.close()

        lf_name = "results/" + dataset + "/mcq/" + name.replace("/", "-") + ".txt"
        lf = open(lf_name, "a")
        lf.write(str(arguments) + "\n\n")
        lf.close()
    else:
        path = training_args.output_dir + "/val_results/"
        Path(path).mkdir(parents=True, exist_ok=True)

    # Metric
    def compute_metrics(eval_predictions):

        global running_set
        global epoch_count

        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        acc = round((preds == label_ids).astype(np.float32).mean().item(), 4)

        if running_set == "validation":
            pickle.dump([predictions, label_ids], open(path + "-val-logits.pkl", "wb"))
            
            val_preds = [str(item) for item in preds]
            with open(path + "-val-preds-epoch-" + str(epoch_count+1) + ".txt", "w") as f:
                f.write("\n".join(list(val_preds)))
                
            if training_args.do_train:
                if epoch_count != 0:
                    x = "Epoch {}: Instance Acc: Val {}".format(epoch_count, acc)
                    lf = open(lf_name, "a")
                    lf.write(x+ "\n\n")
                    lf.close()

                    f = open(fname, "a")
                    f.write(x + "\n\n")
                    f.close()
                epoch_count += 1

        elif running_set == "test":
            test_preds = [str(item) for item in preds]
            with open(path + "-epoch-" + str(epoch_count+1) + ".txt", "w") as f:
                f.write("\n".join(list(test_preds)))
        
        return {"accuracy": acc} 
                

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        running_set = "validation"
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        # epoch_count += 1

    # Test
    if training_args.do_predict:
        running_set = "test"
        logger.info("*** Test ***")

        metrics = trainer.predict(test_dataset)
        # max_test_samples = len(test_dataset)
        # metrics["test_samples"] = len(test_dataset)
        # trainer.log_metrics("test", metrics)
        # trainer.save_metrics("test", metrics)


    kwargs = dict(
        finetuned_from=model_args.model_name_or_path,
        tasks="multiple-choice",
        dataset_tags=dataset,
        dataset_args="regular",
        dataset=dataset.upper(),
        language="en",
    )

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
        
    if training_args.do_train:
        lf = open(lf_name, "a")
        lf.write("-"*100 + "\n")
        lf.close()
        

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
