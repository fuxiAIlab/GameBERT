# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from modeling_utils import PreTrainedModel, Conv1D

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT_FNS = {"relu": nn.ReLU, "swish": swish, "gelu": gelu}


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False, verbose=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.verbose = config.verbose

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)


    def _attn(self, q, k, v, attention_mask=None, head_mask=None):
        w = torch.matmul(q, k)  # [n_batch, n_head, n_seq, n_seq]
        if self.verbose:
            print("  w.shape", w.shape)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # w = w * self.bias + -1e9 * (1 - self.bias)  # TF implem method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        b = self.bias[:, :, : w.size(-2), : w.size(-1)]
        if self.verbose:
            print('  b.shape:', b.shape)
            print('  w.shape:', w.shape)
            # input("  Before w = w * b + - 1e4 * (1 - b), Press Enter to continue...")
        w = w * b + - 1e4 * (1 - b)
        # if verbose:input("  After w = w * b + - 1e4 * (1 - b), Press Enter to continue...")

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask
        # if verbose:input("  After attention mask done.")

        w = nn.Softmax(dim=-1)(w)
        # if verbose:input("  After softmax done. w.shape={}".format(w.shape))

        w = self.attn_dropout(w)
        # if verbose: input("  After dropout")

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask
        if self.verbose:
            # input("  After head mask")
            print("  w.shape", w.shape)
        outputs = torch.matmul(w, v)
        if self.verbose:
            print("  outputs.shape", outputs.shape)
            # input("  After wXv")
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, attention_mask=None, head_mask=None):
        # x: [n_batch, n_seq, n_embd]  [8, 512, 256]
        x = self.c_attn(x)
        # x: [n_batch, n_seq, n_embd * 3 ]  [8, 512, 768]
        # print("  after self.c_attn x.shape", x.shape)
        query, key, value = x.split(self.split_size, dim=2)  # [8， 512， 256]
        query = self.split_heads(query)  # [n_batch, n_head, n_seq, n_embd_per_head]
        key = self.split_heads(key, k=True)  # [n_batch, n_head, n_embd_per_head, n_seq]
        value = self.split_heads(value)  # [n_batch, n_head, n_seq, n_embd_per_head]

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask)
        a = attn_outputs

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = a
        return outputs


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT_FNS[config.afn]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)

    def forward(self, x, attention_mask=None, head_mask=None):
        a = self.attn(x, attention_mask=attention_mask, head_mask=head_mask)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)

        return h


class OpenAIGPTPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


OPENAI_GPT_START_DOCSTRING = r"""    OpenAI GPT model was proposed in
    `Improving Language Understanding by Generative Pre-Training`_
    by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.
    It's a causal (unidirectional) transformer pre-trained using language modeling on a large
    corpus will long range dependencies, the Toronto Book Corpus.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`Improving Language Understanding by Generative Pre-Training`:
        https://openai.com/blog/language-unsupervised/

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.OpenAIGPTConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

OPENAI_GPT_INPUTS_DOCSTRING = r"""    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            GPT is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.
            Indices can be obtained using :class:`transformers.BPT2Tokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            A parallel sequence of tokens (can be used to indicate various portions of the inputs).
            The embeddings from these tokens will be summed with the respective token embeddings.
            Indices are selected in the vocabulary (unlike BERT which has a specific vocabulary for segment indices)
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""

class OpenAIGPTModel(OpenAIGPTPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the last layer of the model.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        model = OpenAIGPTModel.from_pretrained('openai-gpt')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config):
        super(OpenAIGPTModel, self).__init__(config)
        self.verbose = config.verbose
        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        block = Block(config.n_ctx, config, scale=True)

        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])

        self.init_weights()


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if position_ids is None:
            # This was used when we had a single embedding matrice from position and token embeddings
            # start = self.config.vocab_size + self.config.n_special
            # end = start + input_ids.size(-1)
            # position_ids = torch.arange(start, end, dtype=torch.long, device=input_ids.device)
            position_ids = torch.arange(input_ids.size(-1), dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Attention mask.
        if attention_mask is not None:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.tokens_embed(input_ids)
        position_embeds = self.positions_embed(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.tokens_embed(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)
        if self.verbose:
            print("hidden_states.shape", hidden_states.shape)
            # input("Before blocks, Press Enter to continue...")
        for i, block in enumerate(self.h):
            if self.verbose:print("Now in block %s...." % i)
            outputs = block(hidden_states, attention_mask, head_mask[i])
            if self.verbose:
                # input("BLock %s done." % i)
                print("BLock %s done." % i)
            hidden_states = outputs

        outputs = hidden_states.view(*output_shape)
        return outputs  # last hidden state


class OpenAIGPTLMHeadModel(OpenAIGPTPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-1`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        super(OpenAIGPTLMHeadModel, self).__init__(config)
        self.transformer = OpenAIGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.transformer.tokens_embed)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None):
        transformer_outputs = self.transformer(input_ids,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask)
        hidden_states = transformer_outputs
        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,)
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits

