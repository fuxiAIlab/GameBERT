#!/bin/bash

python mlp.py --n_gpu 1 >> /bot-detection/mlp.log 2>&1
echo "bot detection mlp done."

python rnn.py --n_gpu 1 >> /bot-detection/rnn.log 2>&1
echo "bot detection rnn done."

python mlp_encoder.py --n_gpu 4 --model_tag gpt512 >> /bot-detection/gpt512.log 2>&1
echo "bot detection gpt512 done."

python mlp_encoder.py --n_gpu 4 --model_tag bert64 >> /bot-detection/bert64.log 2>&1
echo "bot detection bert64 done."

python mlp_encoder.py --n_gpu 4 --model_tag bert128 >> /bot-detection/bert128.log 2>&1
echo "bot detection bert128 done."

python mlp_encoder.py --n_gpu 4 --model_tag bert256 >> /bot-detection/bert256.log 2>&1
echo "bot detection bert256 done."

python mlp_encoder.py --n_gpu 4 --model_tag bert512 >> /bot-detection/bert512.log 2>&1
echo "bot detection bert512 done."

python mlp_encoder.py --n_gpu 4 --model_tag bert_te >> /bot-detection/bert_te.log 2>&1
echo "bot detection bert_te done."

python mlp_encoder.py --n_gpu 4 --model_tag bert_lss >> /bot-detection/bert_lss.log 2>&1
echo "bot detection bert_lss done."

python mlp_encoder.py --n_gpu 4 --model_tag bert_full >> /bot-detection/bert_full.log 2>&1
echo "bot detection bert_full done."



