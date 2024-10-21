import argparse
import logging
import os
import random
import torch
import numpy as np
import datasets
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from transformers.file_utils import get_full_repo_name
from transformers import AdamW, get_scheduler
from torch.utils.data import DataLoader
from accelerate import Accelerator
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Transformers model on a text classification task")
    parser.add_argument(
        "--pair_train_file", type=str, required=True, default=None,
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--pair_val_file", type=str, required=True, default=None,
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to store the final model.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation on the validation set.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    accelerator = Accelerator()

    # 加载数据集
    data_files = {"train": args.pair_train_file, "validation": args.pair_val_file}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)

    def preprocess_function(examples):
        return tokenizer(examples['text'], padding="max_length", max_length=args.max_seq_length, truncation=True)

    tokenized_datasets = datasets.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = tokenized_datasets["train"]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if args.do_train:
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)

    if args.do_eval:
        eval_results = trainer.evaluate()
        print(eval_results)

if __name__ == "__main__":
    main()
# python run__pairbase.py --model_name_or_path bert-base-uncased --output_dir ./output --pair_train_file  data/splits/altlex_test.csv --pair_val_file data/splits/altlex_train.csv  --max_seq_length 128 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 3 --do_train --do_eval
