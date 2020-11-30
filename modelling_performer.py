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

    def forward(self, x, return_encodings=False, **kwargs):
        b, n, device = *x.shape, x.device
        # token and positional embeddings
        x = self.token_emb(x)
        x += self.pos_emb(torch.arange(n, device=device))
        x = self.dropout(x)

        # performer layers
        x = self.performer(x, **kwargs)[:, 0, :]

        return x


class SiamesePerformer(nn.Module):
    def __init__(self, num_tokens, max_seq_len, dim, depth, heads, local_attn_heads=0, local_window_size=256,
                 causal=False, ff_mult=4, nb_features=None, reversible=False, ff_chunks=1, ff_glu=False, emb_dropout=0.,
                 ff_dropout=0., attn_dropout=0., generalized_attention=False, kernel_fn=nn.ReLU(), qr_uniform_q=False,
                 use_scalenorm=False, use_rezero=False, cross_attend=False):
        super().__init__()
        self.model = PerformerForSiamese(num_tokens, max_seq_len, dim, depth, heads, local_attn_heads,
                                         local_window_size,
                                         causal, ff_mult, nb_features, reversible, ff_chunks, ff_glu, emb_dropout,
                                         ff_dropout, attn_dropout, generalized_attention, kernel_fn, qr_uniform_q,
                                         use_scalenorm, use_rezero, cross_attend)

        self.cosine_similarity = nn.CosineSimilarity()
        self.loss_function = nn.BCELoss()

    def fix_projection_matrix(self):
        self.model.fix_projection_matrices_()

    @torch.no_grad()
    def get_embedding(self, x, mask=None):
        if mask is None:
            mask = torch.ones_like(x).bool()
        return self.model(x, mask)

    def forward(self, x1: LongTensor, x2: LongTensor, target: FloatTensor):
        embedding1 = self.model(x1["input_ids"], mask=x1["attention_mask"].bool())
        embedding2 = self.get_embedding(x2["input_ids"], mask=x2["attention_mask"].bool())

        distance = self.cosine_similarity(embedding1, embedding2)
        loss = self.loss_function(distance, target)

        return loss


if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    model = SiamesePerformer(num_tokens=tokenizer.vocab_size, max_seq_len=512, dim=512, depth=6, heads=8)
    optimizer = Adam(model.parameters())
    sentence1_tensor = tokenizer("Ich bin Andre", add_special_tokens=True, return_tensors="pt")
    sentence2_tensor = tokenizer("Ich bin Peter", add_special_tokens=True, return_tensors="pt")
    loss = model(sentence1_tensor, sentence2_tensor, target=FloatTensor([1]))
    loss.backward()
    optimizer.step()
