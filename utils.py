import argparse
import os
from typing import List

import deepspeed
from torch.utils.data import Dataset
from transformers import AutoTokenizer, EvalPrediction

from preprocessing import SentencePair


class DataLoaderLaper(Dataset):
    def __init__(self, sentence_list: List[SentencePair]):
        self.items = sentence_list

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return {
            "source": self.items[idx].get_source(),
            "target": self.items[idx].get_target()
        }


def run_tensorboard():
    os.system(
        "tensorboard --logdir=" + os.environ.get("LOG_DIR",
                                                 "./logs") + " --reload_interval=15 " + "--port=" + os.environ.get(
            "TENSORBOARD_PORT", "6006") + " --host " + os.environ.get("TENSORBOARD_HOST", "127.0.0.1"))


def add_argument():
    parser = argparse.ArgumentParser(description='enwik8')

    parser.add_argument('--with_cuda', default=True, action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False, action='store_true',
                        help='whether use exponential moving average')
    parser.add_argument('-e', '--epochs', default=int(os.environ.get("EPOCHS")), type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def data_collector_deepspeed(batch_of_sentences, _tokenizer, rank):
    batch_of_sentences = [SentencePair(batch_of_sentences["source"][h], batch_of_sentences["target"][h]) for h in
                          range(len(batch_of_sentences["source"]))]

    source_batch = _tokenizer([s.get_source() for s in batch_of_sentences], add_special_tokens=True, padding=True,
                              return_tensors="pt")
    target_batch = _tokenizer([s.get_target() for s in batch_of_sentences], add_special_tokens=True, padding=True,
                              return_tensors="pt")
    src_in = source_batch["input_ids"].transpose(0, 1)[0:512].transpose(0, 1).to(rank),
    src_attn = source_batch["attention_mask"].transpose(0, 1)[0:512].transpose(0, 1).to(rank)
    tgt_in = target_batch["input_ids"].transpose(0, 1)[0:512].transpose(0, 1).detach().to(rank),
    tgt_attn = target_batch["attention_mask"].transpose(0, 1)[0:512].transpose(0, 1).detach().to(rank)

    return {
        "x1": {
            "input_ids": src_in[0],
            "attention_mask": src_attn
        },
        "x2": {
            "input_ids": tgt_in[0],
            "attention_mask": tgt_attn
        },
    }


tokenizer = AutoTokenizer.from_pretrained(os.environ.get("PRETRAINED_MODEL_AND_TOKENIZER", "distilroberta-base"))


def data_collector_huggingface(batch_of_sentences):
    global tokenizer, rank
    source_batch = tokenizer([s["source"] for s in batch_of_sentences], add_special_tokens=True, padding=True,
                             return_tensors="pt", truncation=True, max_length=512)
    target_batch = tokenizer([s["target"] for s in batch_of_sentences], add_special_tokens=True, padding=True,
                             return_tensors="pt", truncation=True, max_length=512)
    src_in = source_batch["input_ids"]  # .transpose(0, 1)[0:512].transpose(0, 1),
    src_attn = source_batch["attention_mask"]  # .transpose(0, 1)[0:512].transpose(0, 1)
    tgt_in = target_batch["input_ids"]  # .transpose(0, 1)[0:512].transpose(0, 1),
    tgt_attn = target_batch["attention_mask"]  # .transpose(0, 1)[0:512].transpose(0, 1)

    return {
        "x1": {
            "input_ids": src_in,
            "attention_mask": src_attn
        },
        "x2": {
            "input_ids": tgt_in,
            "attention_mask": tgt_attn
        },
    }
