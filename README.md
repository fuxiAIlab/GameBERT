# GameBERT
This repo is the implementation of GameBERT. 
Since the corpus and dataset are too huge we only upload only part of them to run the code.

To run the code, please put the `GameBERT` repo in the `/GameBERT` location. 


## Pretrain Models
In `src` directory, `baseline-bert` and `baseline-gpt` are baselines of traditional pretrained models.
While `BERT-with-timeEmbedding`, `BERT-with-LongSeqSupport` and `BERT-full` are proposed models.

The `BERT` implementation was token from codertimo's [code](https://github.com/codertimo/BERT-pytorch).
And the `GPT` was token from huggingface's [code](https://github.com/huggingface/transformers).

To pretrain the GPT model, please run this script:
```bash
cd /GameBERT/src/baseline-gpt
python run_openai_gpt.py
```

To pretrain the models except GPT, please run these python scripts for
```bash
cd /GameBERT/scr/[current_dir]
python run_pretrain.py
```
Where `current_dir` means `baseline-bert` or `BERT-full` or `BERT-with-timeEmbeding` or `BERT-with-longSeqSupport`.

After pretraining models for specific iterations, 
use the `inference.py` in different methods to produce behaviors feature vectors respectively. 
Like: 
```bash
cd /GameBERT/src/baseline-gpt
python inference.py [model_path_of_pretrained_GPT]
```

## Downstream tasks Evaluation
We provide the sampled dataset for three downstream tasks.
To run the evaluation scripts, please use the `runner.sh` in different downstream tasks.
Like:
```bash
cd /GameBERT/src/downstream-tasks-evaluation/bot-detection
bash runner.sh
``` 