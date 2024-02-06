#!/bin/bash

## Need to set appropriate paths

FILE_GOLD=../../atd-mcl/atd-mcl/full/main/split-118/json/test-all.json
FILE_NAME=`basename $FILE_GOLD`
FILE_PRED1=../eacl2024_experiments/results/CR/rule1/$FILE_NAME
FILE_PRED2=../eacl2024_experiments/results/CR/rule2/$FILE_NAME

cd eval_scripts

## Rule-CR1 (clustering condition: no merge)
poetry run python eval_scripts/coref_evaluator.py \
       -g $FILE_GOLD -gs \
       -p $FILE_PRED1 -ps \
       --no_rename_sentence_id 

## Rule-CR2 (clustering condition: text)
poetry run python eval_scripts/coref_evaluator.py \
       -g $FILE_GOLD -gs \
       -p $FILE_PRED2 -ps \
       --no_rename_sentence_id 
