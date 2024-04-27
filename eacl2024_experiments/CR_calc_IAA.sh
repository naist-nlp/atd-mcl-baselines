#!/bin/bash

INDIR1=../../atd-mcl/atd-mcl/full/agreement/json_per_doc/step2a_coreference/worker1
INDIR2=../../atd-mcl/atd-mcl/full/agreement/json_per_doc/step2a_coreference/worker2

cd eval_scripts

poetry run python eval_scripts/coref_evaluator.py \
       -g $INDIR1 \
       -p $INDIR2 \
       --rename_sentence_id

poetry run python eval_scripts/coref_evaluator.py \
       -g $INDIR1 \
       -p $INDIR2 \
       --rename_sentence_id \
       --ignore_labels LOC_NOM,FAC_NOM,LINE_NOM,TRANS_NOM,LOC_ORG,FAC_ORG,LOC_OR_FAC,DEICTIC

