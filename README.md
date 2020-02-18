# Transformers tutorial

## Purpose
Provide examples of fine tuning Transformers models with HuggingFace Transformers library
Transformers provides several pretrained models : BERT, RoBERTa, GPT etc...

## Pipelines
*transformers* pipelines tasks are described here :
https://github.com/huggingface/transformers#quick-tour-of-pipelines

Useful tasks : feature-extraction, sentiment-analysis etc...
## Install transformers from source (that provides examples scripts)
```bash
git clone https://github.com/huggingface/transformers
```
## Install rust (for tokenizers library)
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
## Install & activate conda env
````bash
./install.sh
````
Activate Anaconda local environment as below:
```bash
conda activate ${PWD}/.conda
```

# How to train/fine tune a transformer model
Copy *run_language_modeling.py* from transformers folder where in easytransformers

You can modify it from training from scratch

I renamed the unmodified script as *run_language_modeling_orig.py*

Here we choose *run_language_modeling.py* script as it is unsupervised and doesn't require labeled data.

## Train from scratch (not recommended)
See https://huggingface.co/blog/how-to-train

You can train a custom tokenizer (with tokenizers library) : see *TokenizerTrainer.py*

```bash
python run_language_modeling.py \
    --output_dir /Users/fanch/PythonProjects/easytransformers/roberta-lm/ \
    --model_type roberta \
    --mlm \
    --tokenizer_name /Users/fanch/PythonProjects/easytransformers/roberta-lm/ \
    --do_train \
    --train_data_file=/Users/fanch/PythonProjects/easytransformers/data/montecristo/montecristo-train.txt \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --save_total_limit 2 \
    --save_steps 2000 \
    --seed 42 \
    --overwrite_output_dir
```

## Fine tune (recommended) from pretrained camemBert model
### With line by line arg
```bash
python run_language_modeling_orig.py \
    --output_dir=montecristo_model \
    --model_type=camembert \
    --model_name_or_path=camembert-base \
    --do_train \
    --train_data_file=/Users/fanch/PythonProjects/easytransformers/data/montecristo/montecristo-train.txt \
    --mlm \
    --line_by_line \
    --overwrite_output_dir
```
### Without line by line arg
```bash
python run_language_modeling_orig.py \
    --output_dir=montecristo_model_2 \
    --model_type=camembert \
    --model_name_or_path=camembert-base \
    --do_train \
    --train_data_file=/Users/fanch/PythonProjects/easytransformers/data/montecristo/montecristo-train.txt \
    --mlm \
    --overwrite_output_dir
```
## Expose with tranformers CLI
Here we use fill-mask task to fill a mask token.

### In python
See [main.py](./main.py)
 
### In bash
```bash
echo -e "Edmond Dantès est le comte de Monte-<mask>." | \
transformers-cli run --task fill-mask \
--model montecristo_model/checkpoint-1000 \
--tokenizer montecristo_model/checkpoint-1000 \
--output test
```

```python
import pickle
test = pickle.load(open("test.pickle", "rb"))
print(test)
```

### REST API
```bash
transformers-cli serve --task fill-mask \
--model montecristo_model/checkpoint-1000 \
--tokenizer montecristo_model/checkpoint-1000
```

See [http://localhost:8888/docs](http://localhost:8888/docs) for Swagger

```bash
curl -X POST "http://localhost:8888/forward" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"inputs\":\"Edmond Dantès est le comte de Monte -<mask>.\"}"
```

## Reads about transformers models
http://jalammar.github.io/illustrated-transformer/
http://jalammar.github.io/illustrated-bert/