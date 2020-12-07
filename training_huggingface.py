import os
import time
import warnings
from datetime import datetime

import torch
from transformers import RobertaTokenizer
from transformers.trainer import Trainer
from transformers.trainer import TrainingArguments
from transformers.trainer_utils import EvaluationStrategy

from lamb import Lamb
from modelling_dual_encoder_performer import DualEncoderPerformer, DualEncoderRoberta
from preprocessing import download_and_extract, Corpus
from utils import DataLoaderLaper, data_collector_huggingface

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = RobertaTokenizer.from_pretrained(os.environ.get("PRETRAINED_VOCAB_PATH", "distilroberta-base"))

warnings.simplefilter("ignore", UserWarning)

tokenizer = RobertaTokenizer.from_pretrained(
    "roberta-large" if not bool(int(os.environ.get("ROBERTA", 1))) else "distilroberta-base")

assert download_and_extract(path=os.environ.get("DATA_DIR", "./storage"))
corpus = Corpus()
corpus.load_corpus(debug=bool(int(os.environ.get("DEBUG", 1))), path=os.environ.get("DATA_DIR", "./storage"))

train_dataset = DataLoaderLaper(
    corpus.get_train() if not bool(int(os.environ.get("DOWNSAMPLE", 1))) else corpus.get_train()[0:100])
test_dataset = DataLoaderLaper(
    corpus.get_dev() if not bool(int(os.environ.get("DOWNSAMPLE", 1))) else corpus.get_dev()[0:100])
print(f"Trainingdata amount {len(train_dataset)}")
auto_encoder = DualEncoderPerformer(tokenizer.vocab_size) if not bool(
    int(os.environ.get("ROBERTA", 1))) else DualEncoderRoberta()

training_args = TrainingArguments(
    output_dir="./results",  # output directory
    num_train_epochs=int(os.environ.get("EPOCHS", 5)),  # total # of training epochs
    per_device_train_batch_size=int(os.environ.get("BATCH_SIZE_PER_GPU", 5)),
    # batch size per device during training
    per_device_eval_batch_size=int(os.environ.get("BATCH_SIZE_PER_GPU", 5)),  # batch size for evaluation
    warmup_steps=int(os.environ.get("WARMUP_NUM_STEPS", 5)),
    save_steps=int(os.environ.get("STEPS_PER_SAVE", 1000000)),
    logging_steps=int(os.environ.get("STEPS_PER_PRINT", 1)),  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./tensorboard',  # directory for storing logs
    evaluation_strategy=EvaluationStrategy.EPOCH,
    save_total_limit=5,
    prediction_loss_only=True,
    gradient_accumulation_steps=int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", 1))
)
optimizer = Lamb(auto_encoder.parameters(), float(os.environ.get("LEARNING_RATE", 0.001)))
lambda1 = lambda epoch: 0.1 * epoch
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
trainer = Trainer(
    model=auto_encoder,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=test_dataset,  # evaluation dataset
    data_collator=data_collector_huggingface,
    optimizers=(optimizer, scheduler)
)
start_time = time.time()
print(f"Starttime {datetime.now()}")
output = trainer.train()
print(f"Endtime {datetime.now()}")
end_time = time.time()
print(
    f"The training took {(end_time - start_time)} seconds = {((end_time - start_time) / 60)} minutes = {((end_time - start_time) / 60 / 60)} hours")
