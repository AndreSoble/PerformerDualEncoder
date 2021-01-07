import os
import time
import warnings
from datetime import datetime

import threading
import torch
from transformers import RobertaTokenizer, AutoTokenizer
from transformers.trainer import Trainer
from transformers.trainer import TrainingArguments
from transformers.trainer_utils import EvaluationStrategy

from lamb import Lamb
from modelling_dual_encoder import DualEncoderPerformer, DualEncoder
from preprocessing import download_and_extract, Corpus
from utils import DataLoaderLaper, data_collector_huggingface, run_tensorboard, CustomTrainer

os.system("rm -r -f /tensorboard/*")
x = threading.Thread(target=run_tensorboard)
x.start()
device = "cuda" if torch.cuda.is_available() else "cpu"

warnings.simplefilter("ignore", UserWarning)

tokenizer = AutoTokenizer.from_pretrained(
    os.environ.get("PRETRAINED_MODEL_AND_TOKENIZER", "distilbert-base-multilingual-cased"))

assert download_and_extract(path=os.environ.get("DATA_DIR", "./storage")) # Could not download and extract training data
corpus = Corpus(downsampled=bool(int(os.environ.get("DOWNSAMPLE", 1))),
                downsampled_count=int(os.environ.get("DOWNSAMPLE_COUNT", 1000)))
corpus.load_corpus(debug=bool(int(os.environ.get("DEBUG", 0))), path=os.environ.get("DATA_DIR", "./storage"))

train_dataset = DataLoaderLaper(corpus.get_train(shuffled=True))
test_dataset = DataLoaderLaper(corpus.get_dev())
eval_dataset = DataLoaderLaper(corpus.get_eval())
print(f"Trainingdata amount {len(train_dataset)}")
auto_encoder = DualEncoderPerformer(tokenizer.vocab_size) if not bool(
    int(os.environ.get("ROBERTA", 1))) else DualEncoder()

training_args = TrainingArguments(
    output_dir="./results",  # output directory
    num_train_epochs=int(os.environ.get("EPOCHS", 10)),  # total # of training epochs
    per_device_train_batch_size=int(os.environ.get("BATCH_SIZE_PER_GPU", 5)),
    # batch size per device during training
    per_device_eval_batch_size=int(os.environ.get("BATCH_SIZE_PER_GPU", 5)),  # batch size for evaluation
    warmup_steps=int(os.environ.get("WARMUP_NUM_STEPS", 5)),
    save_steps=int(os.environ.get("STEPS_PER_SAVE", 1000000)),
    logging_steps=int(os.environ.get("STEPS_PER_PRINT", 1)),  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./tensorboard',  # directory for storing logs
    evaluation_strategy=EvaluationStrategy.STEPS,
    eval_steps=int(os.environ.get("STEPS_PER_SAVE", int(50 / 5))),
    save_total_limit=5,
    prediction_loss_only=True,
    gradient_accumulation_steps=int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", 1)),
    max_grad_norm=0.5
)
optimizer = Lamb(auto_encoder.parameters(), float(os.environ.get("LEARNING_RATE", 5e-3)))
trainer = CustomTrainer(
    model=auto_encoder,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=test_dataset,  # evaluation dataset
    data_collator=data_collector_huggingface,
    optimizers=(optimizer, None)
)

start_time = time.time()
print(f"Starttime {datetime.now()}")
output = trainer.train()
print("Running final evaluation")
trainer.evaluate(eval_dataset)
print(f"Endtime {datetime.now()}")
end_time = time.time()
print(
    f"The training took {(end_time - start_time)} seconds = {((end_time - start_time) / 60)} minutes = {((end_time - start_time) / 60 / 60)} hours")
