#!/bin/bash

INDIR1=../../atd-mcl/atd-mcl/full/agreement/json_per_doc/step2a_coreference/worker1
INDIR2=../../atd-mcl/atd-mcl/full/agreement/json_per_doc/step2a_coreference/worker2

cd eval_scripts

poetry run python eval_scripts/coref_evaluator.py \
       -g $INDIR1 \
       -p $INDIR2

poetry run python eval_scripts/coref_evaluator.py \
       -g $INDIR1 \
       -p $INDIR2 \
       --name_mention_only
