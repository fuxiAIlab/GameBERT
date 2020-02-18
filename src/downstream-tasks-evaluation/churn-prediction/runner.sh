#!/bin/bash

python3 data_preprocess.py 242 empty >> /churn-prediction/dp_242_empty.log 2>&1 && echo "242 empty data process done."
python3 main.py --server 242 --model_tag empty >> /churn-prediction/242_empty.log 2>&1 && echo "242 empty done."
python3 data_preprocess.py 289 empty >> /churn-prediction/dp_289_empty.log 2>&1 && echo "289 empty data process done."
python3 main.py --server 289 --model_tag empty >> /churn-prediction/289_empty.log 2>&1 && echo "289 empty done."

python3 data_preprocess.py 242 gpt512 >> /churn-prediction/dp_242_gpt512.log 2>&1 && echo "242 gpt512 data process done."
python3 main.py --server 242 --model_tag gpt512 >> /churn-prediction/242_gpt512.log 2>&1 && echo "242 gpt512 done."
python3 data_preprocess.py 289 gpt512 >> /churn-prediction/dp_289_gpt512.log 2>&1 && echo "289 gpt512 data process done."
python3 main.py --server 289 --model_tag gpt512 >> /churn-prediction/289_gpt512.log 2>&1 && echo "289 gpt512 done."

python3 data_preprocess.py 242 bert64 >> /churn-prediction/dp_242_bert64.log 2>&1  && echo "242 bert64 data process done."
python3 main.py --server 242 --model_tag bert64 >> /churn-prediction/242_bert64.log 2>&1 && echo "242 bert64 done."
python3 data_preprocess.py 289 bert64 >> /churn-prediction/dp_289_bert64.log 2>&1  && echo "289 bert64 data process done."
python3 main.py --server 289 --model_tag bert64 >> /churn-prediction/289_bert64.log 2>&1 && echo "289 bert64 done."

python3 data_preprocess.py 242 bert128 >> /churn-prediction/dp_242_bert128.log 2>&1  && echo "242 bert128 data process done."
python3 main.py --server 242 --model_tag bert128 >> /churn-prediction/242_bert128.log 2>&1 && echo "242 bert128 done."
python3 data_preprocess.py 289 bert128 >> /churn-prediction/dp_289_bert128.log 2>&1  && echo "289 bert128 data process done."
python3 main.py --server 289 --model_tag bert128 >> /churn-prediction/289_bert128.log 2>&1 && echo "289 bert128 done."

python3 data_preprocess.py 242 bert256 >> /churn-prediction/dp_242_bert256.log 2>&1  && echo "242 bert256 data process done."
python3 main.py --server 242 --model_tag bert256 >> /churn-prediction/242_bert256.log 2>&1 && echo "242 bert256 done."
python3 data_preprocess.py 289 bert256 >> /churn-prediction/dp_289_bert256.log 2>&1  && echo "289 bert256 data process done."
python3 main.py --server 289 --model_tag bert256 >> /churn-prediction/289_bert256.log 2>&1 && echo "289 bert256 done."

python3 data_preprocess.py 242 bert512 >> /churn-prediction/dp_242_bert512.log 2>&1  && echo "242 bert512 data process done."
python3 main.py --server 242 --model_tag bert512 >> /churn-prediction/242_bert512.log 2>&1 && echo "242 bert512 done."
python3 data_preprocess.py 289 bert512 >> /churn-prediction/dp_289_bert512.log 2>&1  && echo "289 bert512 data process done."
python3 main.py --server 289 --model_tag bert512 >> /churn-prediction/289_bert512.log 2>&1 && echo "289 bert512 done."

python3 data_preprocess.py 242 bert_te >> /churn-prediction/dp_242_bert_te.log 2>&1  && echo "242 bert_te data process done."
python3 main.py --server 242 --model_tag bert_te >> /churn-prediction/242_bert_te.log 2>&1 && echo "242 bert_te done."
python3 data_preprocess.py 289 bert_te >> /churn-prediction/dp_289_bert_te.log 2>&1  && echo "289 bert_te data process done."
python3 main.py --server 289 --model_tag bert_te >> /churn-prediction/289_bert_te.log 2>&1 && echo "289 bert_te done."

python3 data_preprocess.py 242 bert_lss >> /churn-prediction/dp_242_bert_lss.log 2>&1  && echo "242 bert_lss data process done."
python3 main.py --server 242 --model_tag bert_lss >> /churn-prediction/242_bert_lss.log 2>&1 && echo "242 bert_lss done."
python3 data_preprocess.py 289 bert_lss >> /churn-prediction/dp_289_bert_lss.log 2>&1  && echo "289 bert_lss data process done."
python3 main.py --server 289 --model_tag bert_lss >> /churn-prediction/289_bert_lss.log 2>&1 && echo "289 bert_lss done."

python3 data_preprocess.py 242 bert_full >> /churn-prediction/dp_242_bert_full.log 2>&1  && echo "242 bert_full data process done."
python3 main.py --server 242 --model_tag bert_full >> /churn-prediction/242_bert_full.log 2>&1 && echo "242 bert_full done."
python3 data_preprocess.py 289 bert_full >> /churn-prediction/dp_289_bert_full.log 2>&1 && echo "289 bert_full data process done."
python3 main.py --server 289 --model_tag bert_full >> /churn-prediction/289_bert_full.log 2>&1 && echo "289 bert_full done."


