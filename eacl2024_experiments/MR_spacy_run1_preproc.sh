#!/bin/bash

## Need to set appropriate paths

DIR_GOLD=../../atd-mcl/atd-mcl/full/main/split-118/json
DIR_GOLD_DB=../../eacl2024_experiments/results/MR/spacy/gold_docbin
MODEL_FOR_SEG=ja_ginza_electra

cd ent_tools/ent_tools_spacy

mkdir -p $DIR_GOLD_DB

poetry run python ent_tools_spacy/json_to_docbin.py \
       -m $MODEL_FOR_SEG \
       -i $DIR_GOLD/train-all.json \
       -o $DIR_GOLD_DB/train.spacy

poetry run python ent_tools_spacy/json_to_docbin.py \
       -m $MODEL_FOR_SEG \
       -i $DIR_GOLD/set-b_dev.json \
       -o $DIR_GOLD_DB/dev.spacy

poetry run python ent_tools_spacy/json_to_docbin.py \
       -m $MODEL_FOR_SEG \
       -i $DIR_GOLD/test-all.json \
       -o $DIR_GOLD_DB/test.spacy
