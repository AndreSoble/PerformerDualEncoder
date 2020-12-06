import os
import warnings
from datetime import time, datetime

import deepspeed
import torch
from transformers import RobertaTokenizer

from modelling_dual_encoder_performer import DualEncoderPerformer
from preprocessing import Corpus, download_and_extract
from utils import DataLoaderLaper, add_argument, data_collector_deepspeed

warnings.simplefilter(action='ignore', category=FutureWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":

    print("Loading data...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    assert download_and_extract(path=os.environ.get("DATA_DIR", "./storage"))
    corpus = Corpus()
    corpus.load_corpus(debug=bool(int(os.environ.get("DEBUG", 1))), path=os.environ.get("DATA_DIR", "./storage"))

    train_dataset = DataLoaderLaper(
        corpus.get_train() if not bool(int(os.environ.get("DOWNSAMPLE", 1))) else corpus.get_train()[0:10000])

    auto_encoder = DualEncoderPerformer(tokenizer.vocab_size).cuda()

    cmd_args = add_argument()
    model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=cmd_args, model=auto_encoder,
                                                                   model_parameters=auto_encoder.parameters(),
                                                                   training_data=train_dataset)

    for epoch in range(int(os.environ.get("EPOCHS", 1))):
        if model_engine.local_rank == 0:
            print(f"{datetime.now()} Epoch {epoch}")
        losses = list()
        for i, data in enumerate(trainloader):
            model_engine.train()
            data = data_collector_deepspeed(data, tokenizer, model_engine.local_rank)
            loss = model_engine(**data)[0]
            loss = loss.mean()
            losses.append(loss.item())
            model_engine.backward(loss)
            model_engine.step()
            if model_engine.local_rank != 0:
                continue

            if ((i * epoch + i) % int(os.environ.get("STEPS_PER_PRINT")) == 0 or i == (len(trainloader)-1)) and i != 0:
                print(f"{datetime.now()} Epoch {epoch} iter {i} Loss {sum(losses) / len(losses)}")
                model_engine.save_checkpoint(os.environ.get("OUTPUT_DIR"), (i * epoch + i))
        if model_engine.local_rank != 0:
            continue
        print(f"{datetime.now()} Epoch {epoch} final epoch Loss {sum(losses) / len(losses)} on rank 0")
    if model_engine.local_rank == 0:
        auto_encoder.fix_projection_matrix()
        auto_encoder.save_pretrained(os.environ.get("OUTPUT_DIR") + "/final_performer.bin")
