import argparse
import collections
import os
import traceback
from random import shuffle
from typing import List, Optional, Dict, Union, Any

import deepspeed
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.trainer import Trainer, logger

from preprocessing import SentencePair


class CustomTrainer(Trainer):

    def _report_to_hp_search(
            self, trial: Union["optuna.Trial", Dict[str, Any]], epoch: int, metrics: Dict[str, float]
    ):
        pass

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            class PredictionOutput(NamedTuple):
                predictions: Union[np.ndarray, Tuple[np.ndarray]]
                label_ids: Optional[np.ndarray]
                metrics: Optional[Dict[str, float]]
        """

        model = self.model

        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        batch_size = eval_dataloader.batch_size
        print("***** Running %s *****", "Evaluation")

        logger.info("***** Running %s *****", "Evaluation")
        logger.info("  Batch size = %d", batch_size)
        print("1")
        model.eval()

        losses = list()
        true_losses = list()
        for step, inputs in enumerate(tqdm(eval_dataloader)):
            try:
                print("2")
                with torch.no_grad():
                    print("2.1")
                    inputs = self._prepare_inputs(inputs)
                    print("2.2", inputs["x1"]["input_ids"].size(), inputs["x1"]["attention_mask"].size(),
                          inputs["x2"]["input_ids"].size(), inputs["x2"]["attention_mask"].size())
                    # todo 7 Segmentation fault      (core dumped) python /workspace/training_huggingface.py an diesen Punkt
                    outputs = model(**inputs)
                    print("2.3")
                    true_similarities = torch.nn.functional.cosine_similarity(outputs[1], outputs[2])
                    print("2.4")
                    true_diff = torch.ones_like(true_similarities) - true_similarities
                    print("2.5")
                    true_loss = torch.mean(true_diff).item()
                    print("3")
                    N = outputs[1].size()[0]
                    neg = list()
                    for i in range(N):
                        xxx = torch.zeros(N - 1).to(outputs[1].device)
                        negative_samples_similarities_exp = [
                            torch.nn.functional.cosine_similarity(outputs[1][i].unsqueeze(0),
                                                                  outputs[2][n].unsqueeze(0))
                            for n in
                            range(N) if n != i]
                        for idx in range(N - 1):
                            xxx[idx] = negative_samples_similarities_exp[idx]
                        neg.append(torch.mean(xxx).item())
                    print("4")
                    true_loss1 = sum(neg) / len(neg) + true_loss
                    losses.append(outputs[0].mean().item())
                    true_losses.append(true_loss1)
            except Exception:
                print("5")
                print(traceback.print_exc())
        print("6")
        metrics = {
            "understandable_loss": sum(true_losses) / len(true_losses),
            "loss": sum(losses) / len(losses)
        }
        print("7")
        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)
        print("8")
        self.log(metrics)
        print("9")
        return metrics


class DataLoaderLaper(Dataset):
    def __init__(self, sentence_list: List[SentencePair], shuffle_every_epoch=False):
        self.items = sentence_list
        self.shuffle_every_epoch = shuffle_every_epoch

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if idx == 0 and self.shuffle_every_epoch:
            shuffle(self.items)

        return {
            "source": self.items[idx].get_source(),
            "target": self.items[idx].get_target()
        }


def run_tensorboard():
    os.system(
        "tensorboard --logdir=" + os.environ.get("LOG_DIR",
                                                 "./tensorboard") + " --port=6006 --host=0.0.0.0")


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
