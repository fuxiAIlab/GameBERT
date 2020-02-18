import torch.nn as nn

from .transformer import TransformerBlock
from .embedding import BERTEmbedding


class BERT(nn.Module):

    def __init__(self, vocab_size, max_seq_len, hidden, n_layers, attn_heads, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.feed_forward_hidden = hidden * 4

        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, max_seq_len=max_seq_len)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        # attention masking for padded token
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embedding(x)

        # print("\n")
        for idx, transformer in enumerate(self.transformer_blocks):
            # print("in transformer block:", idx)
            x = transformer.forward(x, mask)

        return x
