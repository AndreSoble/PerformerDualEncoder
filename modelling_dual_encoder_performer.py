import os

import torch
from performer_pytorch.performer_pytorch import cast_tuple, find_modules, FastAttention, get_module_device, Performer
from torch import nn
from torch.nn.modules.loss import _Loss
from transformers import AutoModel, AutoTokenizer


class PerformerForDualEncoder(nn.Module):
    def __init__(self, num_tokens, max_seq_len, dim, depth, heads, local_attn_heads=0, local_window_size=256,
                 causal=False, ff_mult=4, nb_features=None, reversible=False, ff_chunks=1, ff_glu=False, emb_dropout=0.,
                 ff_dropout=0., attn_dropout=0., generalized_attention=False, kernel_fn=nn.ReLU(), qr_uniform_q=False,
                 use_scalenorm=False, use_rezero=False, cross_attend=False):
        super().__init__()
        local_attn_heads = cast_tuple(local_attn_heads)

        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(emb_dropout)

        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        self.performer = Performer(dim, depth, heads, local_attn_heads, local_window_size, causal, ff_mult, nb_features,
                                   reversible, ff_chunks, generalized_attention, kernel_fn, qr_uniform_q, use_scalenorm,
                                   use_rezero, ff_glu, ff_dropout, attn_dropout, cross_attend)
        self.linear = nn.Linear(10, 10)
        self.activation = torch.nn.Softsign()

    def fix_projection_matrices_(self):
        fast_attentions = find_modules(self, FastAttention)
        device = get_module_device(self)
        for fast_attention in fast_attentions:
            fast_attention.set_projection_matrix(device)

    def forward(self, x, mask):
        b, n, device = *x.shape, x.device
        # token and positional embeddings
        x = self.token_emb(x)
        x += self.pos_emb(torch.arange(n, device=device))
        x = self.dropout(x)

        # performer layers
        x = self.performer(x, mask=mask)  # [:, 0, :]

        x = x.mean(1)

        x = self.activation(x)

        return x


class AMSLoss(_Loss):
    def __init__(self, m=0.3):
        super(AMSLoss, self).__init__()
        self.margin = m
        self.cosine_similarity = nn.CosineSimilarity()

    def rank(self, x: torch.FloatTensor, y: torch.FloatTensor):

        N = x.size()[0]
        ret = torch.zeros(N).to(x.device)
        similarities = self.cosine_similarity(x, y)

        for i in range(N):
            xxx = torch.zeros(N - 1).to(x.device)
            negative_samples_similarities_exp = [self.cosine_similarity(x[i].unsqueeze(0), y[n].unsqueeze(0)) for n in
                                                 range(N) if n != i]
            for idx in range(N - 1):
                xxx[idx] = negative_samples_similarities_exp[idx]
            negative_samples_similarities_exp = torch.exp(xxx)
            negative_samples_similarities_exp = torch.sum(negative_samples_similarities_exp)
            m1 = torch.exp(torch.sub(similarities[i], self.margin))
            m2 = torch.exp(torch.sub(similarities[i], self.margin))
            ret[i] = torch.div(m1, torch.add(m2, negative_samples_similarities_exp))

        return torch.mul(-1 / N, torch.sum(ret))

    def forward(self, x: torch.FloatTensor, y: torch.FloatTensor, one_direction=False):
        if not one_direction:
            return torch.add(self.rank(x, y), self.rank(y, x))  # self.rank(x, y)#
        else:
            return self.rank(x, y)


class DualEncoderPerformer(nn.Module):
    def __init__(self, num_tokens, max_seq_len=2048, dim=512, depth=6, heads=8, local_attn_heads=0,
                 local_window_size=256,
                 causal=False, ff_mult=4, nb_features=None, reversible=False, ff_chunks=10, ff_glu=False,
                 emb_dropout=0.1,
                 ff_dropout=0.1, attn_dropout=0.1, generalized_attention=False, kernel_fn=nn.ReLU(), qr_uniform_q=False,
                 use_scalenorm=False, use_rezero=False, cross_attend=False):
        super().__init__()
        self.vocab_size = num_tokens
        self.model = PerformerForDualEncoder(num_tokens, max_seq_len, dim, depth, heads, local_attn_heads,
                                             local_window_size,
                                             causal, ff_mult, nb_features, reversible, ff_chunks, ff_glu, emb_dropout,
                                             ff_dropout, attn_dropout, generalized_attention, kernel_fn, qr_uniform_q,
                                             use_scalenorm, use_rezero, cross_attend)

    def fix_projection_matrix(self):
        self.model.fix_projection_matrices_()

    @torch.no_grad()
    def get_embedding(self, x, mask=None):
        if mask is None:
            mask = torch.ones_like(x).bool().detach()
        return self.model(x, mask)

    def forward(self, x1: dict, x2: dict):
        embedding1 = self.model(x1["input_ids"].to(self.model.linear.weight.device),
                                mask=x1["attention_mask"].to(self.model.linear.weight.device).bool())
        embedding2 = self.model(x2["input_ids"].to(self.model.linear.weight.device),
                                mask=x2["attention_mask"].to(self.model.linear.weight.device).bool())
        loss_function = AMSLoss()
        return (loss_function(embedding1, embedding2, one_direction=False), embedding1, embedding2)

    @torch.no_grad()
    def get_similarity(self, x1: dict, x2: dict):
        x1_emb = self.get_embedding(x1["input_ids"], mask=x1["attention_mask"].bool())
        x2_emb = self.get_embedding(x2["input_ids"], mask=x2["attention_mask"].bool())
        return torch.nn.functional.cosine_similarity(x1_emb, x2_emb)

    def save_pretrained(self, path):
        torch.save({"vocab_size": self.vocab_size,
                    "states": self.state_dict()}, path)

    @staticmethod
    def from_pretrained(path):
        si = torch.load(path)
        cls = DualEncoderPerformer(si["vocab_size"])
        cls.load_state_dict(si["states"])


class DualEncoderRoberta(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(os.environ.get("PRETRAINED_MODEL_AND_TOKENIZER", "distilroberta-base"))

    def fix_projection_matrix(self):
        pass

    @torch.no_grad()
    def get_embedding(self, x, mask=None):
        if mask is None:
            mask = torch.ones_like(x).detach()
        return self.model(x, attention_mask=mask)[1]

    def forward(self, x1: dict, x2: dict):
        loss_function = AMSLoss()
        embedding1 = self.model(x1["input_ids"],
                                attention_mask=x1["attention_mask"])[1]
        embedding2 = self.model(x2["input_ids"],
                                attention_mask=x2["attention_mask"])[1]
        return (loss_function(embedding1, embedding2, one_direction=False), embedding1, embedding2)

    @torch.no_grad()
    def get_similarity(self, x1: dict, x2: dict):
        x1_emb = self.get_embedding(x1["input_ids"], mask=x1["attention_mask"])
        x2_emb = self.get_embedding(x2["input_ids"], mask=x2["attention_mask"])
        return torch.nn.functional.cosine_similarity(x1_emb, x2_emb)

    def save_pretrained(self, path):
        torch.save({"vocab_size": self.vocab_size,
                    "states": self.state_dict()}, path)

    @staticmethod
    def from_pretrained(path):
        si = torch.load(path)
        cls = DualEncoderPerformer(si["vocab_size"])
        cls.load_state_dict(si["states"])


if __name__ == "__main__":
    from fastai.optimizer import Lamb

    # tokenizer = RobertaTokenizerFast.from_pretrained(
    #    "roberta-large" if not bool(int(os.environ.get("ROBERTA"))) else "distilroberta-base")
    # model = DualEncoderPerformer(num_tokens=tokenizer.vocab_size, max_seq_len=512, dim=512, depth=6, heads=8)



    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    model = DualEncoderPerformer(tokenizer.vocab_size)
    optimizer = Lamb(model.parameters(), lr=0.001)  # Lamb
    sentence1_tensor = tokenizer(["Ich bin Andre", "Ich brauche hilfe", "Du magst tanzen?"],
                                 add_special_tokens=True, return_tensors="pt",
                                 padding=True)
    sentence2_tensor = tokenizer(["I am Andre", "I need support", "do you like dancing?"],
                                 add_special_tokens=True, return_tensors="pt",
                                 padding=True)

    sentence1_test = tokenizer(["Ich bin Andre", "Ich bin Andre"],
                               add_special_tokens=True, return_tensors="pt",
                               padding=True)
    sentence2_test = tokenizer(["I am Andre", "I need support"],
                               add_special_tokens=True, return_tensors="pt",
                               padding=True)

    for _ in range(200):
        loss = model(sentence1_tensor, sentence2_tensor)[0]
        print(loss.item())
        loss.backward()
        optimizer.step()
        print(model.get_similarity(sentence1_test, sentence2_test))
