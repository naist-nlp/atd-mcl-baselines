#!/bin/bash

## Need to set appropriate paths

DIR_FINAL=../eacl2024_experiments/results/ED/bert
FILE_PRED_NAME=test-all.ed_results.json
FILE_PRED=$DIR_FINAL/$FILE_PRED_NAME
FILE_EVAL_NAME=test-all.ed_results.eval.json
FILE_EVAL=$DIR_FINAL/$FILE_EVAL_NAME

#python ed_bert/convert_json_format_for_eval.py \
#       -i $FILE_PRED \
#       -o $FILE_EVAL

DB_PATH=../data/osm/20230620_all_extnames.txt
FILE_GOLD=../../../atd-mcl/atd-mcl/full/main/split-118/json/test-all.json

cd eval_scripts || exit

poetry run python eval_scripts/ed_evaluator_for_groups.py \
       -d $DB_PATH \
       -g $FILE_GOLD \
       -p $FILE_EVAL
