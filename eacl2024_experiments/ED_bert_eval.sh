#!/bin/bash

## Need to set appropriate paths

DB_PATH=../data/osm/20230620_all_extnames.txt
FILE_GOLD=../../atd-mcl/atd-mcl/full/main/split-118/json/test-all.json
FILE_NAME=`basename $FILE_GOLD`
FILE_PRED=

cd eval_scripts

poetry run python eval_scripts/ed_evaluator_for_groups.py \
       -d $DB_PATH \
       -g $FILE_GOLD \
       -p $FILE_PRED \
       --use_orig_ent_id
