#!/bin/bash

## Need to set appropriate paths

DB_PATH=../data/osm/20230620_all_extnames.txt
DIR_FINAL=eacl2024_experiments/results/ED/bert
DIR_PROGRESS=ed_bert/results

mkdir -p $DIR_FINAL
mkdir -p $DIR_PROGRESS

FILE_GOLD=../../atd-mcl/atd-mcl/full/main/split-118/json/test-all.json
ENT_NAME_FILE=$DIR_PROGRESS/test-all.names.longest.txt

python ed_bert/ed_select_entity_names.py \
       -i "$FILE_GOLD" \
       -o "$ENT_NAME_FILE"

DIR_BERT_MODEL=ed_bert/model/cl-tohoku/bert-base-japanese-whole-word-masking
mkdir -p $DIR_BERT_MODEL
cd $DIR_BERT_MODEL || exit
wget https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/vocab.txt
wget https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/config.json
wget https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/tokenizer_config.json
wget https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/pytorch_model.bin

python ed_bert/convert_text_to_vectors.py \
       -i "$ENT_NAME_FILE"