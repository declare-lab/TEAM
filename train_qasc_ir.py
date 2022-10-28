import json
import time
import random
import pickle
import gc, os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from models import Model
from transformers import get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import Adafactor, get_scheduler

from sklearn.metrics import accuracy_score

class QASCIRDataset(Dataset):
    def __init__(self, f1, f2, sep_token, input_format, shuffle):
        content, labels = [], []
        x = open(f1).readlines()
        
        df = pd.read_csv(f2, sep="\t", names=["context", "answer"])
        ir = [df.iloc[j]["context"].split("\\n")[2] for j in range(len(df))]
        
        if shuffle:
            y = list(zip(x, ir))
            random.shuffle(y)
            x, ir = zip(*y)
        
        for line, ir_context in zip(x, ir):
            instance = json.loads(line)
            question = instance["question"]["stem"]
            choices = [item["text"] for item in instance["question"]["choices"]]
            if "answerKey" in instance:
                l = instance["answerKey"]
            else:
                l = "A"
            
            if input_format == "0":
                for c in choices:
                    content.append("{} {} {} {} {}".format(question, sep_token, c, sep_token, ir_context))
            elif input_format == "1":
                for c in choices:
                    content.append("Question: {} {} Answer: {} {} Context: {}".format(question, sep_token, c, sep_token, ir_context))
            
            answers = ["A", "B", "C", "D", "E", "F", "G", "H"]
            y = [0, 0, 0, 0, 0, 0, 0, 0]
            y[answers.index(l)] = 1
            labels += y
                
        self.content, self.labels = content, labels
        
    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        s1, s2 = self.content[index], self.labels[index]
        return s1, s2
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
    
    
def configure_dataloaders(sep_token="\\n", train_batch_size=16, eval_batch_size=16, shuffle=False, input_format="1"):
    "Prepare dataloaders"
    train_dataset = QASCIRDataset("data/qasc/train.jsonl", "data/qasc_ir/train.tsv", sep_token, input_format, True)
    train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=train_batch_size, collate_fn=train_dataset.collate_fn)

    val_dataset = QASCIRDataset("data/qasc/dev.jsonl", "data/qasc_ir/dev.tsv", sep_token, input_format, False)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=eval_batch_size, collate_fn=val_dataset.collate_fn)
    
    test_dataset = QASCIRDataset("data/qasc/test.jsonl", "data/qasc_ir/test.tsv", sep_token, input_format, False)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=eval_batch_size, collate_fn=val_dataset.collate_fn)

    return train_loader, val_loader, test_loader


def configure_optimizer(model, args):
    """Prepare optimizer and schedule (linear warmup and decay)"""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.wd,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    return optimizer

def configure_scheduler(optimizer, num_training_steps, args):
    "Prepare scheduler"
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps > 0
        else math.ceil(num_training_steps * args.warmup_ratio)
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )    
    return lr_scheduler


def train_or_eval_model(model, dataloader, optimizer=None, split="Train"):
    losses, preds, preds_cls, labels_cls,  = [], [], [], []
    if split=="Train":
        model.train()
    else:
        model.eval()
    
    for batch in tqdm(dataloader, leave=False):
        if split=="Train":
            optimizer.zero_grad()
            
        content, l_cls = batch
        loss, p, p_cls = model(batch)
        
        preds.append(p)
        preds_cls.append(p_cls)
        labels_cls.append(l_cls)
        
        if split=="Train":
            loss.backward()
            optimizer.step()
            
        losses.append(loss.item())

    avg_loss = round(np.mean(losses), 4)
    
    if split=="Train":
        wandb.log({"Train Loss": avg_loss})
        all_preds_cls = [item for sublist in preds_cls for item in sublist]
        all_labels_cls = [item for sublist in labels_cls for item in sublist]
        acc = round(accuracy_score(all_labels_cls, all_preds_cls), 4)
        wandb.log({"Train CLS Accuracy": acc})
        
        return avg_loss, acc
    
    elif split=="Val":
        wandb.log({"Val Loss": avg_loss})
        all_preds_cls = [item for sublist in preds_cls for item in sublist]
        all_labels_cls = [item for sublist in labels_cls for item in sublist]
        acc = round(accuracy_score(all_labels_cls, all_preds_cls), 4)
        wandb.log({"Val CLS Accuracy": acc})
        
        instance_preds = [item for sublist in preds for item in sublist]
        instance_labels = np.array(all_labels_cls).reshape(-1, args.num_choices).argmax(1)
        instance_acc = round(accuracy_score(instance_labels, instance_preds), 4)
        wandb.log({"Val Instance Accuracy": instance_acc})
        
        return avg_loss, acc, instance_acc
    
    elif "Test" in split:
        mapper = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}
        instance_preds = [item for sublist in preds for item in sublist]
        instance_preds = [mapper[item] for item in instance_preds]
        print ("Test preds frequency:", dict(pd.Series(instance_preds).value_counts()))

        return instance_preds
    
    
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate for transformers.")
    parser.add_argument("--wd", default=0.0, type=float, help="Weight decay for transformers.")
    parser.add_argument("--warm-up-steps", type=int, default=0, help="Warm up steps.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float, help="Epsilon for AdamW optimizer.")
    parser.add_argument("--bs", type=int, default=8, help="Batch size.")
    parser.add_argument("--eval-bs", type=int, default=8, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs.")
    parser.add_argument("--name", default="roberta-large", help="Which model.")
    parser.add_argument('--shuffle', action='store_true', default=False, help="Shuffle train data such that positive and negative \
        sequences of the same question are not necessarily in the same batch.")
    parser.add_argument('--input-format', default="1", help="How to format the input data.")
    
    global args
    args = parser.parse_args()
    print(args)
    
    train_batch_size = args.bs
    eval_batch_size = args.eval_bs
    epochs = args.epochs
    name = args.name
    shuffle = args.shuffle
    input_format = args.input_format
    
    num_choices = 8
    vars(args)["num_choices"] = num_choices
    assert eval_batch_size%num_choices == 0, "Eval batch size should be a multiple of num choices, which is 8 for QASC"
    
    model = Model(
        name=name,
        num_choices=num_choices
    ).cuda()
    
    sep_token = model.tokenizer.sep_token
    
    optimizer = configure_optimizer(model, args)
    
    if "/" in name:
        sp = name[name.index("/")+1:]
    else:
        sp = name
    
    exp_id = str(int(time.time()))
    vars(args)["exp_id"] = exp_id
    rs = "Acc: {}"
    
    path = "saved/qasc_ir/" + exp_id + "/" + name.replace("/", "-")
    Path("saved/qasc_ir/" + exp_id + "/").mkdir(parents=True, exist_ok=True)
    
    fname = "saved/qasc_ir/" + exp_id + "/" + "args.txt"
    
    f = open(fname, "a")
    f.write(str(args) + "\n\n")
    f.close()
        
    Path("results/qasc_ir/").mkdir(parents=True, exist_ok=True)
    lf_name = "results/qasc_ir/" + name.replace("/", "-") + ".txt"
    lf = open(lf_name, "a")
    lf.write(str(args) + "\n\n")
    lf.close()

    wandb.init(project="QASC-IR-" + sp)
    wandb.watch(model)
    
    
    for e in range(epochs):
        
        train_loader, val_loader, test_loader = configure_dataloaders(
            sep_token, train_batch_size, eval_batch_size, shuffle, input_format
        )     
        
        train_loss, train_acc = train_or_eval_model(model, train_loader, optimizer, "Train")
        val_loss, val_acc, val_ins_acc = train_or_eval_model(model, val_loader, split="Val")
        test_preds = train_or_eval_model(model, test_loader, split="Test")
        
        with open(path + "-epoch-" + str(e+1) + ".txt", "w") as f:
            f.write("\n".join(list(test_preds)))
        
        x = "Epoch {}: Loss: Train {}; Val {}".format(e+1, train_loss, val_loss)
        y = "Classification Acc: Train {}; Val {}".format(train_acc, val_acc)
        z = "Instance Acc: Val {}".format(val_ins_acc)
            
        print (x)
        print (y)
        print (z)

        lf = open(lf_name, "a")
        lf.write(x + "\n" + y + "\n" + z + "\n\n")
        lf.close()

        f = open(fname, "a")
        f.write(x + "\n" + y + "\n" + z + "\n\n")
        f.close()
        
    lf = open(lf_name, "a")
    lf.write("-"*100 + "\n")
    lf.close()