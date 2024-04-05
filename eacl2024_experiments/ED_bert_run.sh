#!/bin/bash

cd ed_bert || exit

DIR_PROGRESS=results_temp

mkdir -p $DIR_PROGRESS

FILE_GOLD=../../../atd-mcl/atd-mcl/full/main/split-118/json/test-all.json
ENT_NAME_FILE=$DIR_PROGRESS/test-all.names.longest.txt

poetry run python ed_select_entity_names.py \
       -i $FILE_GOLD \
       -o $ENT_NAME_FILE

DIR_BERT_MODEL=model_temp/cl-tohoku/bert-base-japanese-whole-word-masking
mkdir -p $DIR_BERT_MODEL
wget https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/vocab.txt
mv vocab.txt $DIR_BERT_MODEL
wget https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/config.json
mv config.json $DIR_BERT_MODEL
wget https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/tokenizer_config.json
mv tokenizer_config.json $DIR_BERT_MODEL
wget https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/pytorch_model.bin
mv pytorch_model.bin $DIR_BERT_MODEL

poetry run python convert_text_to_vectors.py \
       -i $ENT_NAME_FILE \
       -bert $DIR_BERT_MODEL

DB_PATH=../data/osm/20230620_all_extnames.txt
DB_JSONL_FILE=$DIR_PROGRESS/20230620_all_extnames.jsonl

poetry run python convert_extnames_txt_to_jsonl.py \
       -i $DB_PATH \
       -o $DB_JSONL_FILE

poetry run python convert_extnames_jsonl_to_names.py \
       -i $DB_JSONL_FILE

poetry run python convert_extnames_jsonl_to_ids.py \
       -i $DB_JSONL_FILE

DB_NAME_FILE=$DIR_PROGRESS/20230620_all_extnames.names.txt

poetry run python convert_text_to_vectors.py \
       -i $DB_NAME_FILE

ENT_VEC_FILE=$DIR_PROGRESS/test-all.names.longest.subwords.bert-base-japanese-whole-word-masking.vecs.hdf5
DB_ID_FILE=$DIR_PROGRESS/20230620_all_extnames.ids.txt
DB_VEC_FILE=$DIR_PROGRESS/20230620_all_extnames.names.subwords.bert-base-japanese-whole-word-masking.vecs.hdf5
RESULT=$DIR_PROGRESS/test-all.ed_results.json

poetry run python ed_disambiguate_entities.py \
       -i $FILE_GOLD -iv $ENT_VEC_FILE \
       -ei $DB_ID_FILE -ev $DB_VEC_FILE \
       -o $RESULT

DIR_FINAL=../eacl2024_experiments/results/ED/bert
mkdir -p $DIR_FINAL

FILE_PRED_NAME=test-all.ed_results.json
FILE_PRED=$DIR_FINAL/$FILE_PRED_NAME
FILE_EVAL_NAME=test-all.ed_results.eval.json
FILE_EVAL=$DIR_FINAL/$FILE_EVAL_NAME

poetry run python convert_json_format_for_eval.py \
       -i $FILE_PRED \
       -o $FILE_EVAL
