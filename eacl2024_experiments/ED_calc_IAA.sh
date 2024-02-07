#!/bin/bash

INDIR1=../../atd-mcl/atd-mcl/full/agreement/json_per_doc/step2b_link/worker1
INDIR2=../../atd-mcl/atd-mcl/full/agreement/json_per_doc/step2b_link/worker2
URLMAP=../data/agreement/url_map_step2b.txt

cd eval_scripts

## This is not necessary: generate an URL map from practically same URL to aggregated URLs
# poetry run python eval_scripts/aggregate_practically_same_URLs_for_IAA.py \
#        -i ../data/agreement/20230519_step2b_annotators_disagreement_judged.txt \
#        -o $URLMAP

## (a) original URL
poetry run python eval_scripts/ed_evaluator_for_entries.py \
       -g $INDIR1 \
       -p $INDIR2 \
       --use_orig_ent_id \
       --rename_sentence_id

## (b) grouped URL
poetry run python eval_scripts/ed_evaluator_for_entries.py \
       -g $INDIR1 \
       -p $INDIR2 \
       -u $URLMAP \
       --use_orig_ent_id \
       --rename_sentence_id
