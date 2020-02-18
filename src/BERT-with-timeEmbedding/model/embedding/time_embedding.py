import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """
    Mercer's time encoding
    see https://openreview.net/pdf?id=r1gIASreLH
    Self-attention with Functional Time Representation Learning
    """
    def __init__(self, embed_size, expand_dim):
        super(TimeEmbedding, self).__init__()

        self.embed_size = embed_size
        self.expand_dim = expand_dim

        self.period_var = nn.Parameter(torch.linspace(0, 8, embed_size))

        # 3 种实现方式，把常量搬到GPU上。
        # self.expand_coef = (torch.arange(0.0, expand_dim) + 1).reshape([1, -1]).cuda()
        # self.expand_coef = nn.Parameter((torch.arange(0.0, expand_dim) + 1).reshape([1, -1]), requires_grad=False)
        self.register_buffer("expand_coef", (torch.arange(0.0, expand_dim) + 1).reshape([1, -1]))

        basis_expan_var = torch.empty(embed_size, 2*expand_dim)
        nn.init.xavier_uniform_(basis_expan_var)
        self.basis_expan_var = nn.Parameter(basis_expan_var)
        self.basis_expan_var_bias = nn.Parameter(torch.zeros(embed_size))

    def forward(self, inputs):
        # assert torch.sum(torch.isnan(inputs)) == 0
        # todo: how to deal with it when padding with 0's and actually 0， 处理padding为0和真实为0的区别
        # todo: 现在是把min设置为1，有个偏移量

        # print("self.period_var.device", self.period_var.device)
        period_var = (10 ** self.period_var).unsqueeze(1).repeat([1, self.expand_dim])
        # print("period_var.device", period_var.device)
        # print("self.expand_coef.device", self.expand_coef.device)

        freq_var = self.expand_coef / period_var  # [embed_size, expand_dim]

        expand_input = inputs.unsqueeze(-1).repeat([1, 1, self.embed_size])
        # print("expand_input.device", expand_input.device)
        # print("freq_var.device", freq_var.device)

        sin_enc = torch.sin(expand_input.unsqueeze(-1).mul(freq_var.unsqueeze(0).unsqueeze(0)))
        cos_enc = torch.cos(expand_input.unsqueeze(-1).mul(freq_var.unsqueeze(0).unsqueeze(0)))
        time_enc = torch.cat([sin_enc, cos_enc], dim=-1).mul(self.basis_expan_var.unsqueeze(0).unsqueeze(0))
        time_enc = torch.sum(time_enc, dim=-1) + self.basis_expan_var_bias.unsqueeze(0).unsqueeze(0)
        return time_enc


if __name__ == "__main__":
    embed_size = 3
    expand_dim = 20
    te = TimeEmbedding(embed_size, expand_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    te = te.to(device)
    inputs = torch.arange(50).reshape([5, 10]).float()  # [batch_size, max_seq_len]
    time_enc = te(inputs.to(device))  # [batch_size, max_seq_len, embed_size]
    print(time_enc.shape)
