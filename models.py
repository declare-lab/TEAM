import json
import random
import numpy as np
import torch
import transformers
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Model(nn.Module):
    def __init__(
        self,
        name: str,
        num_choices: int
    ):
        super().__init__()
        
        self.name = name
        self.num_choices = num_choices
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(name)
        
        self.max_length = 512
        
        if "base" in name:
            self.hidden_size = 768
        elif "xx-large" in name:
            self.hidden_size = 1536
        elif "large" in name:
            self.hidden_size = 1024
        elif "tiny" in name:
            self.hidden_size = 128
        elif "small" in name:
            self.hidden_size = 768
        elif "aristo" in name:
            self.hidden_size = 768
            
        self.ce_loss_func = nn.CrossEntropyLoss()
        self.scorer = nn.Linear(self.hidden_size, 1)
        
    def score_input(self, content):
        batch = self.tokenizer(
            content, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        out = self.model(
            batch["input_ids"].to(self.model.device), batch["attention_mask"].to(self.model.device),
            output_hidden_states=True
        )
        return out["logits"]

    def forward(self, batch):
        content, labels  = batch
        logits = self.score_input(content)    
        labels = torch.tensor(labels, dtype=torch.long).to(logits.device)
        loss = self.ce_loss_func(logits, labels)
        preds_cls = list(torch.argmax(logits, 1).cpu().numpy())
        positive_probability = logits[:, 1]
        
        preds = torch.argmax(positive_probability.reshape(-1, self.num_choices), 1)
        preds = list(preds.cpu().numpy())
        return loss, preds, preds_cls
    