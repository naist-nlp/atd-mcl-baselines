#!/bin/bash

INDIR1=../atd-mcl/atd-mcl/full/agreement/json_per_doc/step1_mention/worker1
INDIR2=../atd-mcl/atd-mcl/full/agreement/json_per_doc/step1_mention/worker2
DOCIDS="00019,00036,00265,00351,00401,00545,01018,01026,01059,01106"

cd eval_scripts
poetry run python eval_scripts/ner_evaluator.py \
       -g $INDIR1 \
       -p $INDIR2 \
       -d $DOCIDS
