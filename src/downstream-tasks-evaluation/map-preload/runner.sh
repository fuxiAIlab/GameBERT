#!/bin/bash

python3 data_preprocess.py 242 gpt512
python3 data_preprocess.py 204 gpt512

python3 data_preprocess.py 242 bert64
python3 data_preprocess.py 204 bert64

python3 data_preprocess.py 242 empty
python3 data_preprocess.py 204 empty

python3 data_preprocess.py 242 bert128
python3 data_preprocess.py 204 bert128

python3 data_preprocess.py 242 bert256
python3 data_preprocess.py 204 bert256

python3 data_preprocess.py 242 bert512
python3 data_preprocess.py 204 bert512

python3 data_preprocess.py 242 bert_te
python3 data_preprocess.py 204 bert_te

python3 data_preprocess.py 242 bert_lss
python3 data_preprocess.py 204 bert_lss

python3 data_preprocess.py 242 bert_full
python3 data_preprocess.py 204 bert_full

python3 -u main.py --server 242 --model_tag empty >> /map-preload-prediction/242_empty.log 2>> /map-preload-prediction/errlog && echo "242 empty done."
python3 -u main.py --server 204 --model_tag empty >> /map-preload-prediction/204_empty.log 2>> /map-preload-prediction/errlog && echo "204 empty done."
python3 -u main.py --server 242 --model_tag gpt512 >> /map-preload-prediction/242_gpt512.log 2>> /map-preload-prediction/errlog && echo "242 gpt512 done."
python3 -u main.py --server 204 --model_tag gpt512 >> /map-preload-prediction/204_gpt512.log 2>> /map-preload-prediction/errlog && echo "204 gpt512 done."
python3 -u main.py --server 242 --model_tag bert64 >> /map-preload-prediction/242_bert64.log 2>> /map-preload-prediction/errlog && echo "242 bert64 done."
python3 -u main.py --server 204 --model_tag bert64 >> /map-preload-prediction/204_bert64.log 2>> /map-preload-prediction/errlog && echo "204 bert64 done."
python3 -u main.py --server 242 --model_tag bert128 >> /map-preload-prediction/242_bert128.log 2>> /map-preload-prediction/errlog && echo "242 bert128 done."
python3 -u main.py --server 204 --model_tag bert128 >> /map-preload-prediction/204_bert128.log 2>> /map-preload-prediction/errlog && echo "204 bert128 done."
python3 -u main.py --server 242 --model_tag bert256 >> /map-preload-prediction/242_bert256.log 2>> /map-preload-prediction/errlog && echo "242 bert256 done."
python3 -u main.py --server 204 --model_tag bert256 >> /map-preload-prediction/204_bert256.log 2>> /map-preload-prediction/errlog && echo "204 bert256 done."
python3 -u main.py --server 242 --model_tag bert512 >> /map-preload-prediction/242_bert512.log 2>> /map-preload-prediction/errlog && echo "242 bert512 done."
python3 -u main.py --server 204 --model_tag bert512 >> /map-preload-prediction/204_bert512.log 2>> /map-preload-prediction/errlog && echo "204 bert512 done."
python3 -u main.py --server 242 --model_tag bert_te >> /map-preload-prediction/242_bert_te.log 2>> /map-preload-prediction/errlog && echo "242 bert_te done."
python3 -u main.py --server 204 --model_tag bert_te >> /map-preload-prediction/204_bert_te.log 2>> /map-preload-prediction/errlog && echo "204 bert_te done."
python3 -u main.py --server 242 --model_tag bert_lss >> /map-preload-prediction/242_bert_lss.log 2>> /map-preload-prediction/errlog && echo "242 bert_lss done."
python3 -u main.py --server 204 --model_tag bert_lss >> /map-preload-prediction/204_bert_lss.log 2>> /map-preload-prediction/errlog && echo "204 bert_lss done."
python3 -u main.py --server 242 --model_tag bert_full >> /map-preload-prediction/242_bert_full.log 2>> /map-preload-prediction/errlog && echo "242 bert_full done."
python3 -u main.py --server 204 --model_tag bert_full >> /map-preload-prediction/204_bert_full.log 2>> /map-preload-prediction/errlog && echo "204 bert_full done."