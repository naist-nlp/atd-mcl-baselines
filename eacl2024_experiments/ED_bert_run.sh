#!/bin/bash

## Need to set appropriate paths

DB_PATH=../data/osm/20230620_all_extnames.txt
DIR_PRED=eacl2024_experiments/results/ED/bert

FILE_GOLD=../../atd-mcl/atd-mcl/full/main/split-118/json/test-all.json
ENT_NAME_FILE=$DIR_PRED/test-all.names.longest.txt

mkdir -p $DIR_PRED

python ed_bert/ed_select_entity_names.py \
       -i "$FILE_GOLD" \
       -o "$ENT_NAME_FILE"

