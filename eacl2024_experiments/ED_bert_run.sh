#!/bin/bash

## Need to set appropriate paths

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

DB_PATH=data/osm/20230620_all_extnames.txt
DB_JSONL_FILE=$DIR_PROGRESS/20230620_all_extnames.jsonl

python ed_bert/convert_extnames_txt_to_jsonl.py \
       -i "$DB_PATH" \
       -o "$DB_JSONL_FILE"

python ed_bert/convert_extnames_jsonl_to_names.py \
       -i "$DB_JSONL_FILE"

python ed_bert/convert_extnames_jsonl_to_ids.py \
       -i "$DB_JSONL_FILE"

DB_NAME_FILE=$DIR_PROGRESS/20230620_all_extnames.names.txt

python ed_bert/convert_text_to_vectors.py \
       -i "$DB_NAME_FILE"

ENT_VEC_FILE=$DIR_PROGRESS/test-all.names.longest.subwords.bert-base-japanese-whole-word-masking.vecs.hdf5
DB_ID_FILE=$DIR_PROGRESS/20230620_all_extnames.ids.txt
DB_VEC_FILE=$DIR_PROGRESS/20230620_all_extnames.names.subwords.bert-base-japanese-whole-word-masking.vecs.hdf5
RESULT=$DIR_PROGRESS/test-all.ed_results.json

python ed_bert/ed_disambiguate_entities.py \
       -i "$FILE_GOLD" -iv "$ENT_VEC_FILE" \
       -ei "$DB_ID_FILE" -ev "$DB_VEC_FILE" \
       -o "$RESULT"
