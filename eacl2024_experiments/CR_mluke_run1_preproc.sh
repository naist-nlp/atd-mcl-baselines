#!/bin/bash

## Need to set appropriate paths

DIR_GOLD=../../atd-mcl/atd-mcl/full/main/split-118/json
DIR_GOLD_OUTPUT=../../eacl2024_experiments/results/CR/mluke/gold

cd data_preprocessor/mluke_tools

mkdir -p $DIR_GOLD_OUTPUT

poetry run python mluke_tools/convert_atd_to_cr_jsonl.py \
    $DIR_GOLD/train-all.json \
    $DIR_GOLD_OUTPUT/train.jsonl

poetry run python mluke_tools/convert_atd_to_cr_jsonl.py \
    $DIR_GOLD/set-b_dev.json \
    $DIR_GOLD_OUTPUT/dev.jsonl

poetry run python mluke_tools/convert_atd_to_cr_jsonl.py \
    $DIR_GOLD/test-all.json \
    $DIR_GOLD_OUTPUT/test.jsonl
