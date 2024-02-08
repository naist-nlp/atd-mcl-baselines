#!/bin/bash

## Need to set appropriate paths

FILE_GOLD=../atd-mcl/atd-mcl/full/main/split-118/json/test-all.json
FILE_PRED=../eacl2024_experiments/results/MR/spacy/json/test-all.json

cd eval_scripts

poetry run python eval_scripts/ner_evaluator.py \
       -g $FILE_GOLD \
       -p $FILE_PRED
