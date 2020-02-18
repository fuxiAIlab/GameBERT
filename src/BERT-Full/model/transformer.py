import torch.nn as nn
import torch
import torch.nn.functional as F
from math import sqrt
from .utils import SublayerConnection, PositionwiseFeedForward


class SparseAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(SparseAttention, self).__init__()
        self.query_transform = nn.Linear(embed_size, embed_size)
        # self.key_transform = nn.Linear(embed_size, embed_size)
        self.value_transform = nn.Linear(embed_size, embed_size)

        self.post_att = nn.Linear(embed_size, embed_size)

        self.num_heads = num_heads
        self.embed_size = embed_size
        self.d_k = self.embed_size // self.num_heads
        assert embed_size % num_heads == 0, "embed_size must be divided by num_heads, " \
                                            "embed_size={}, num_heads={}".format(embed_size, num_heads)

    def forward(self, x, pos_mask):
        # x: [batch_size, seq_len, embed_size]
        B, T, E = x.shape  # [B, T, E]
        _, W, _ = pos_mask.shape  # [T, W, T]
        H = self.num_heads

        # input("before attend inputs")  # 1047MiB, 799MiB
        # version 2
        attend_inputs = torch.matmul(x.transpose(1, 2), pos_mask.view(T, -1))  # [B, E, W*T]
        try:
            attend_inputs = attend_inputs.contiguous().view(B, E, W, T).permute(0, 3, 2, 1)
        except:
            print("\n")
            print("x.shape", x.shape)
            print("position_mask.shape", pos_mask.shape)
            print("position_mask.view(T, -1).shape", pos_mask.view(T, -1).shape)
            print("attend_inputs.shape", attend_inputs.shape)
            print(B, E, W, T)
            raise
        # print("attend_inputs.shape", attend_inputs.shape)  # 8, 1024, 32, 256 [B,T,W,E]

        # input("after attend inputs:")  # 1799MiB, 1055MiB

        query = self.query_transform(x).contiguous().view(B, T, 1, self.d_k, H).permute(0, 4, 1, 2, 3)  # [B, H, T, 1, dk]
        key = self.query_transform(attend_inputs).contiguous().view(B, T, W, self.d_k, H).permute(0, 4, 1, 2, 3)  # [B, H, T, W, dk]
        value = self.value_transform(attend_inputs).contiguous().view(B, T, W, self.d_k, H).permute(0, 4, 1, 2, 3)  # [B, H, T, W, dk]

        # input("after q,k,v linear transform")  # 4815MiB, 2087MiB

        scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(self.embed_size)  # [B, H, T, 1, W]

        # input("after get outputs")  # 5623MiB, 2367MiB

        scores = F.softmax(scores, dim=-1)

        outputs = torch.matmul(scores, value)  # [B, H, T, 1, dk]
        outputs = outputs.squeeze(3).transpose(1, 2).reshape([B, T, E])

        # input("after get final outputs")  # 6471MiB, 2663MiB

        return self.post_att(outputs)


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = SparseAttention(num_heads=attn_heads, embed_size=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, pos_mask):
        # assert torch.sum(torch.isnan(x)) == 0, x
        # assert torch.sum(torch.isnan(mask)) == 0
        # x = [batch_size, seq_len, d_model]
        x = self.input_sublayer(x, lambda _x: self.attention(_x, pos_mask=pos_mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
