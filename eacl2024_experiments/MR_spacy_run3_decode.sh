#!/bin/bash

## Need to set appropriate paths

MODEL=../../eacl2024_experiments/results/MR/spacy/model_atd-mcl_split-118/model-best
DIR_GOLD_DB=../../eacl2024_experiments/results/MR/spacy/gold_docbin
DIR_PRED=../../eacl2024_experiments/results/MR/spacy/json

cd ent_tools/ent_tools_spacy

mkdir -p $DIR_PRED

poetry run python ent_tools_spacy/decode.py \
       -m $MODEL \
       -i $DIR_GOLD_DB/test.spacy \
       -o $DIR_PRED/test-all.json
