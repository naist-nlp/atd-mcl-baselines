#!/bin/bash

cd eval_scripts || exit

## Need to set appropriate paths
DB_PATH=../data/osm/20230620_all_extnames.txt
FILE_GOLD=../../../atd-mcl/atd-mcl/full/main/split-118/json/test-all.json
DIR_FINAL=../eacl2024_experiments/results/ED/bert
FILE_EVAL_NAME=test-all.ed_results.eval.json
FILE_EVAL=$DIR_FINAL/$FILE_EVAL_NAME

poetry run python eval_scripts/ed_evaluator_for_groups.py \
       -d $DB_PATH \
       -g $FILE_GOLD \
       -p $FILE_EVAL
