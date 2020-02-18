import torch.nn as nn
import torch
import torch.nn.functional as F
from math import sqrt
from .utils import SublayerConnection, PositionwiseFeedForward


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        # query = [batch_size, attn_heads, seq_len, d_k]
        # scores = [batch_size, attn_heads, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / sqrt(query.size(-1))

        # mask = [batch_size, 1, seq_len, seq_len]
        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, -1e9)

        # p_attn = [batch_size, 1, seq_len, seq_len]
        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h  # attn_heads

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query, key, value = [batch_size, attn_heads, seq_len, d_k]
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l in self.linear_layers]

        # 2) Apply attention on all the projected vectors in batch.
        # x = [batch_size, attn_heads, seq_len, d_k]
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # x = [batch_size, seq_len, d_model]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


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

    def forward(self, x, position_mask):
        # x: [batch_size, seq_len, embed_size]
        B, T, E = x.shape  # [B, T, E]
        _, W, _ = position_mask.shape  # [T, W, T]
        H = self.num_heads

        # input("before attend inputs")  # 1047MiB, 799MiB
        # version 2
        attend_inputs = torch.matmul(x.transpose(1, 2), position_mask.view(T, -1))  # [B, E, W*T]
        try:
            attend_inputs = attend_inputs.contiguous().view(B, E, W, T).permute(0, 3, 2, 1)
        except:
            print("\n")
            print("x.shape", x.shape)
            print("position_mask.shape", position_mask.shape)
            print("position_mask.view(T, -1).shape", position_mask.view(T, -1).shape)
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



#
# class SparseAttention(nn.Module):
#     def __init__(self, embed_size, num_heads):
#         super(SparseAttention, self).__init__()
#         self.query_transform = nn.Linear(embed_size, embed_size)
#         self.key_transform = nn.Linear(embed_size, embed_size)
#         self.value_transform = nn.Linear(embed_size, embed_size)
#
#         self.post_att = nn.Linear(embed_size, embed_size)
#
#         self.num_heads = num_heads
#         self.embed_size = embed_size
#         assert embed_size % num_heads == 0, "embed_size must be divided by num_heads, " \
#                                             "embed_size={}, num_heads={}".format(embed_size, num_heads)
#
#     def forward(self, x, position_mask):
#         # x: [batch_size, seq_len, embed_size]
#         batch_size, seq_len, embed_size = x.shape
#         _, width, _ = position_mask.shape
#
#         # todo: version 1
#         # pos_mask_mat: [batch_size, seq_len, width, seq_len]
#         # pos_mask_mat = position_mask.unsqueeze(0).repeat([batch_size, 1, 1, 1])
#
#         # attend_inputs: [batch_size, seq_len, width, embed_size]
#         # todo: x.unsqueeze(1).repeat([1, seq_len, 1, 1]) cost huge memory!!!
#         # attend_inputs = torch.matmul(pos_mask_mat, x.unsqueeze(1).repeat([1, seq_len, 1, 1]))
#
#         # todo: versiom 2
#         position_mask = position_mask.view(-1, seq_len)
#         attend_inputs = torch.matmul(position_mask, x.transpose(0, 1).contiguous().view(seq_len, -1))
#         attend_inputs = attend_inputs.contiguous().view(seq_len, width, batch_size, embed_size).permute(2, 0, 1, 3)
#
#         # query = [batch_size, seq_len, 1, embed_size]
#         query = self.query_transform(x.unsqueeze(2))
#         # key value = [batch_size, seq_len, width, embed_size]
#         key = self.key_transform(attend_inputs)
#         value = self.value_transform(attend_inputs)
#
#         # todo: 这部分先split再concatenate是重新申请了内存，所以消耗很大，应该不如 原始的实现中利用view来达成，不做显示的拆分&合并
#         # q_: [batch_size * self.num_heads, seq_len, 1, embed_size // self.num_heads]
#         q_ = torch.cat(torch.split(query, embed_size // self.num_heads, dim=-1), 0)
#         # k_ v_ = [batch_size * self.num_heads, seq_len, width, embed_size // self.num_heads]
#         k_ = torch.cat(torch.split(key, embed_size // self.num_heads, dim=-1), 0)
#         v_ = torch.cat(torch.split(value, embed_size // self.num_heads, dim=-1), 0)
#
#         # outputs: [batch_size * self.num_heads, seq_len, 1, width]
#         outputs = torch.matmul(q_, k_.permute((0, 1, 3, 2))) / sqrt(self.embed_size)
#
#         # key masking
#         key_masks = torch.sign(torch.abs(torch.sum(key, dim=-1)))  # [batch_size, seq_len, width]
#         key_masks = key_masks.repeat([self.num_heads, 1, 1]).unsqueeze(2)  # [batch_size * num_heads, seq_len, 1, width]
#
#         outputs.masked_fill(key_masks == 0, -2 ** 32 + 1)
#
#         outputs = F.softmax(outputs, dim=-1)
#
#         outputs = torch.matmul(outputs, v_)  # [batch_size * num_heads, seq_len, 1, embed_size // num_heads]
#         outputs = torch.cat(torch.split(outputs, self.num_heads, dim=0), -1)
#         outputs = outputs.reshape([batch_size, seq_len, embed_size])
#
#         return self.post_att(outputs)


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
        self.attention = SparseAttention(embed_size=hidden, num_heads=attn_heads)
        # self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, pos_mask):
        # x = [batch_size, seq_len, d_model]
        x = self.input_sublayer(x, lambda _x: self.attention(_x, pos_mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
