#!/bin/bash

## Need to set appropriate paths

FILE_GOLD=../atd-mcl/atd-mcl/full/main/split-118/json/test-all.json
FILE_PRED=../eacl2024_experiments/results/CR/mluke/pred/test-all.json

cd eval_scripts

poetry run python eval_scripts/coref_evaluator.py \
       -g $FILE_GOLD -gs \
       -p $FILE_PRED -ps
