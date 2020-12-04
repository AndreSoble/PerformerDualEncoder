import os

import torch
from transformers import RobertaTokenizer
from transformers.trainer import Trainer
from transformers.trainer import TrainingArguments
from transformers.trainer_utils import EvaluationStrategy

from modelling_siamese_performer import SiamesePerformer
from preprocessing import download_and_extract, Corpus
from utils import DataLoaderLaper, data_collector_huggingface

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = RobertaTokenizer.from_pretrained(os.environ.get("PRETRAINED_VOCAB_PATH", "distilroberta-base"))

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

assert download_and_extract(path=os.environ.get("DATA_DIR", "./storage"))
corpus = Corpus()
corpus.load_corpus(debug=bool(int(os.environ.get("DEBUG", 1))), path=os.environ.get("DATA_DIR", "./storage"))

train_dataset = DataLoaderLaper(
    corpus.get_train() if not bool(int(os.environ.get("DOWNSAMPLE", 1))) else corpus.get_train()[0:5000])
test_dataset = DataLoaderLaper(
    corpus.get_dev() if not bool(int(os.environ.get("DOWNSAMPLE", 1))) else corpus.get_dev()[0:5000])

auto_encoder = SiamesePerformer(tokenizer.vocab_size)

training_args = TrainingArguments(
    output_dir="./results",  # output directory
    num_train_epochs=20,  # total # of training epochs
    per_device_train_batch_size=100,
    # batch size per device during training
    per_device_eval_batch_size=100,  # batch size for evaluation
    warmup_steps=500,
    save_steps=500,
    logging_steps=100,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    evaluation_strategy=EvaluationStrategy.EPOCH,
    save_total_limit=5,
    prediction_loss_only=True
)

trainer = Trainer(
    model=auto_encoder,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=test_dataset,  # evaluation dataset
    data_collator=data_collector_huggingface
)

output = trainer.train()
