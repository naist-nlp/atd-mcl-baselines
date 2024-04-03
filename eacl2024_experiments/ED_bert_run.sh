#!/bin/bash

## Need to set appropriate paths

DB_PATH=../data/osm/20230620_all_extnames.txt
FILE_GOLD=../../atd-mcl/atd-mcl/full/main/split-118/json/test-all.json
FILE_NAME=`basename $FILE_GOLD`
DIR_PRED=../eacl2024_experiments/results/ED/rule
FILE_PRED=$DIR_PRED/$FILE_NAME

cd rule_based

mkdir -p $DIR_PRED

poetry run python src/ed_select_entity_names.py

poetry run python ed_bert/entity_disambiguator.py \
       -d $DB_PATH \
       -i $FILE_GOLD \
       -o $FILE_PRED
