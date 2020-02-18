import torch.nn as nn
import torch
from .bert import BERTLSS


class BERTLMLSS(nn.Module):
    """
    BERT Language Model with Time Embedding
    Time prediction + Masked Language Model
    """

    def __init__(self, bert: BERTLSS, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super(BERTLMLSS, self).__init__()
        self.bert = bert
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, token_seq, pos_mask):
        x = self.bert(token_seq, pos_mask)
        return self.mask_lm(x)


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
        # assert torch.sum(torch.isnan(x)) == 0, x
        return self.softmax(self.linear(x))
