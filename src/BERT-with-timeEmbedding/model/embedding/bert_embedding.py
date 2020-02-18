import torch.nn as nn
from .token_embedding import TokenEmbedding
from .position_embedding import PositionalEmbedding
from .time_embedding import TimeEmbedding
import torch


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, expand_dim, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.time = TimeEmbedding(embed_size=self.token.embedding_dim, expand_dim=expand_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, token_seq, time_seq):
        # assert torch.sum(torch.isnan(token_seq)) == 0
        # assert torch.sum(torch.isnan(time_seq)) == 0
        x = self.token(token_seq) + self.position(token_seq) + self.time(time_seq)
        return self.dropout(x)
