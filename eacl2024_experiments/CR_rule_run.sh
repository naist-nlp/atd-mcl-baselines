#!/bin/bash

## Need to set appropriate paths

FILE_GOLD=../../atd-mcl/atd-mcl/full/main/split-118/json/test-all.json
FILE_NAME=`basename $FILE_GOLD`
FILE_PRED1=../eacl2024_experiments/results/CR/rule1/$FILE_NAME
FILE_PRED2=../eacl2024_experiments/results/CR/rule2/$FILE_NAME

cd rule_based

## Rule-CR1 (clustering condition: no merge)
poetry run python rule_based/coreference_resolver.py \
       -i $FILE_GOLD \
       -o $FILE_PRED1 \
       --no_merge

## Rule-CR2 (clustering condition: text)
poetry run python rule_based/coreference_resolver.py \
       -i $FILE_GOLD \
       -o $FILE_PRED2
