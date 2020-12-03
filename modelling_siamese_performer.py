import torch
from performer_pytorch.performer_pytorch import cast_tuple, find_modules, FastAttention, get_module_device, Performer
from torch import nn, LongTensor, FloatTensor
from torch.optim import Adam
from transformers import RobertaTokenizer


class PerformerForSiamese(nn.Module):
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
        return x


class AMSLoss:
    def __init__(self, m=0.3):
        self.margin = m
        self.cosine_similarity = nn.CosineSimilarity()

    def rank(self, x: torch.FloatTensor, y: torch.FloatTensor):

        N = x.size()[0]
        ret = torch.empty(N).to(x.device)
        similarities = self.cosine_similarity(x, y)

        for i in range(N):
            xxx = torch.empty(N - 1).to(x.device)
            negative_samples_similarities_exp = [self.cosine_similarity(x[i].unsqueeze(0), y[n].unsqueeze(0)) for n in
                                                 range(N) if n != i]
            for idx in range(N - 1):
                xxx[idx] = negative_samples_similarities_exp[idx]
            negative_samples_similarities_exp = torch.exp(xxx)
            negative_samples_similarities_exp = torch.sum(negative_samples_similarities_exp)
            m1 = torch.exp(torch.sub(similarities[i], self.margin))
            m2 = torch.exp(torch.sub(similarities[i], self.margin))
            ret[i] = -1 * torch.log(torch.div(m1, torch.add(m2, negative_samples_similarities_exp)))

        return torch.mul(1 / N, torch.sum(ret))

    def calculate_loss(self, x: torch.FloatTensor, y: torch.FloatTensor):
        return torch.add(self.rank(x, y), self.rank(y, x))


class SiamesePerformer(nn.Module):
    def __init__(self, num_tokens, max_seq_len=2048, dim=512, depth=3, heads=4, local_attn_heads=0,
                 local_window_size=256,
                 causal=False, ff_mult=4, nb_features=None, reversible=False, ff_chunks=10, ff_glu=False,
                 emb_dropout=0.1,
                 ff_dropout=0.1, attn_dropout=0.1, generalized_attention=False, kernel_fn=nn.ReLU(), qr_uniform_q=False,
                 use_scalenorm=False, use_rezero=False, cross_attend=False):
        super().__init__()
        self.model = PerformerForSiamese(num_tokens, max_seq_len, dim, depth, heads, local_attn_heads,
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

    def forward(self, x1: LongTensor, x2: LongTensor):
        embedding1 = self.model(x1["input_ids"], mask=x1["attention_mask"].bool())
        embedding2 = self.get_embedding(x2["input_ids"], mask=x2["attention_mask"].bool())
        loss_function = AMSLoss()
        return loss_function.calculate_loss(embedding1, embedding2)


if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    model = SiamesePerformer(num_tokens=tokenizer.vocab_size, max_seq_len=512, dim=512, depth=6, heads=8)
    optimizer = Adam(model.parameters())
    sentence1_tensor = tokenizer(["Ich bin Andre", "Ich bin nicht Andre", "Ich bin ein Student"],
                                 add_special_tokens=True, return_tensors="pt",
                                 padding=True)
    sentence2_tensor = tokenizer(["Ich bin Peter", "Ich bin nicht Peter", "Ich bin kein Student"],
                                 add_special_tokens=True, return_tensors="pt",
                                 padding=True)
    loss = model(sentence1_tensor, sentence2_tensor)
    loss.backward()
    optimizer.step()
