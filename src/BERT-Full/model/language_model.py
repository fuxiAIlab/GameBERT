import torch.nn as nn
import torch
from .bert import BERTFull


class BERTLMFull(nn.Module):
    """
    BERT Language Model with Time Embedding
    Time prediction + Masked Language Model
    """

    def __init__(self, bert: BERTFull, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super(BERTLMFull, self).__init__()
        self.bert = bert
        self.mask_tp = MaskedTimePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, token_seq, time_seq, pos_mask):
        # print("===" * 20)
        # print("token_seq", token_seq)
        # print("===" * 20)
        # print("time_seq", time_seq)
        x = self.bert(token_seq, time_seq, pos_mask)
        # print("===" * 20)
        # print("bert output", x)
        mask_tp_output, mask_lm_output = self.mask_tp(x), self.mask_lm(x)
        return mask_tp_output, mask_lm_output


class MaskedTimePrediction(nn.Module):
    """
    predicting origin time from masked input sequence
    regression task
    """
    def __init__(self, hidden):
        super(MaskedTimePrediction, self).__init__()
        self.linear = nn.Linear(hidden, 1)

    def forward(self, x):
        assert torch.sum(torch.isnan(x)) == 0, x
        # [n_batchsize, seq_len, d_model] -> [n_batch_size, seq_len]
        return torch.relu(self.linear(x).squeeze(-1))


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super(MaskedLanguageModel, self).__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        assert torch.sum(torch.isnan(x)) == 0, x
        return self.softmax(self.linear(x))
