import os
import warnings

import deepspeed
import torch
from transformers import RobertaTokenizer

from modelling_siamese_performer import SiamesePerformer
from preprocessing import Corpus, download_and_extract
from utils import DataLoaderLaper, add_argument, data_collector_deepspeed

warnings.simplefilter(action='ignore', category=FutureWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    print("Loading data...")

    assert download_and_extract(path=os.environ.get("DATA_DIR", "./storage"))
    corpus = Corpus()
    corpus.load_corpus(debug=bool(int(os.environ.get("DEBUG",0))),path=os.environ.get("DATA_DIR", "./storage"))

    tokenizer = RobertaTokenizer.from_pretrained(os.environ.get("PRETRAINED_VOCAB_PATH", "roberta-base"))
    tokenizer.max_len = 1024
    auto_encoder = SiamesePerformer(tokenizer.vocab_size).cuda()

    train_dataset = DataLoaderLaper(corpus.get_train())

    cmd_args = add_argument()
    model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=cmd_args, model=auto_encoder,
                                                                   model_parameters=auto_encoder.parameters(),
                                                                   training_data=train_dataset)

    for epoch in range(int(os.environ.get("EPOCHS", 1))):
        if model_engine.local_rank == 0:
            print(f"Epoch {epoch}")

        for i, data in enumerate(trainloader):
            model_engine.train()
            data = data_collector_deepspeed(data, tokenizer, model_engine.local_rank)
            loss = model_engine(**data)
            loss = loss.mean()
            loss = torch.div(loss,
                             int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", 100)))  # gradient_accumulation_steps
            model_engine.backward(loss)
            model_engine.step()
            if model_engine.local_rank != 0:
                continue

            if (i * epoch + i) % int(os.environ.get("STEPS_PER_PRINT")) == 0:
                model_engine.save_checkpoint(os.environ.get("OUTPUT_DIR"), (i * epoch + i))

    if model_engine.local_rank == 0:
        auto_encoder.fix_projections()
        auto_encoder.save_pretrained(os.environ.get("OUTPUT_DIR") + "/final_performer.bin")
