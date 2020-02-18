import torch
import torch.nn as nn
from .transformer import TransformerBlock
from .embedding import BERTEmbedding


class BERTFull(nn.Module):
    """
    BERT with Time Embedding

    """

    def __init__(self, vocab_size, expand_dim, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, expand_dim=expand_dim)
        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, token_seq, time_seq, pos_mask):

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(token_seq, time_seq)

        # running over multiple transformer blocks
        for idx, transformer in enumerate(self.transformer_blocks):
            # print("transformer block idx:", idx)
            x = transformer.forward(x, pos_mask)

        return x
