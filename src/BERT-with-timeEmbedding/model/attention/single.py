import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        # assert torch.sum(torch.isnan(query)) == 0
        # assert torch.sum(torch.isnan(key)) == 0
        # assert torch.sum(torch.isnan(value)) == 0

        # query = [batch_size, attn_heads, seq_len, d_k]
        # scores = [batch_size, attn_heads, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        # mask = [batch_size, 1, seq_len, seq_len]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # p_attn = [batch_size, 1, seq_len, seq_len]
        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
