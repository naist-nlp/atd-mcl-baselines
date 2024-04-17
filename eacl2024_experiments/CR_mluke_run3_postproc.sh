#!/bin/bash

## Need to set appropriate paths

MODEL_DIR=../../eacl2024_experiments/results/CR/mluke/model_atd-mcl_split-118
DIR_GOLD=../../atd-mcl/atd-mcl/full/main/split-118/json
DIR_PRED=../../eacl2024_experiments/results/CR/mluke/pred

cd data_preprocessor/mluke_tools

mkdir -p $DIR_PRED

poetry run python mluke_tools/convert_cr_jsonl_to_atd.py -p \
    $MODEL_DIR/test_predictions.jsonl \
    $DIR_PRED/test-all.json
