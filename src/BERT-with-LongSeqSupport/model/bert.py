import torch
import torch.nn as nn
from .transformer import TransformerBlock
from .embedding import BERTEmbedding


class BERTLSS(nn.Module):
    """
    BERT with Time Embedding

    """

    def __init__(self, vocab_size, max_seq_len, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
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
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, max_seq_len=max_seq_len)
        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, token_seq, pos_mask):
        # assert torch.sum(torch.isnan(token_seq)) == 0
        # assert torch.sum(torch.isnan(time_seq)) == 0
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (token_seq > 0).unsqueeze(1).repeat(1, token_seq.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(token_seq)
        if len(pos_mask.shape) == 4:
            pos_mask = pos_mask.squeeze(0)

        # running over multiple transformer blocks
        # print("\n")
        for idx, transformer in enumerate(self.transformer_blocks):
            # input("transformer block idx:%s" % idx)
            x = transformer.forward(x, pos_mask)

        return x
