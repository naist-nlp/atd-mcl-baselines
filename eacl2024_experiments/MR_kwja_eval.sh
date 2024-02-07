#!/bin/bash

## Need to set appropriate paths

DIR_GOLD=../atd-mcl/atd-mcl/full/main/mention_tsv_per_doc/set-b
DIR_PRED=../eacl2024_experiments/results/MR/kwja/tsv
LABELMAP=../data/label_conversion_map/label_map_irex_to_atd-mcl.json
DOCIDS="00036,00265,00351,01059,01106,01563,01573,01575,01581,01591,02012,02053,02274,02819,03285,03417,03697,04416,04919,05412,06025,06984,07149,07205,07279,07528,07530,07559,07725,07860,07883,08966,09033,09034,09049,09139,09591,09673,09725,09726,09996,10575,10727,11421,11773,12430,12615,12785,13190,13706,14084,15003,15009,15488,17585,17640,17728,18249,19078,19531,20688,20691,20697,20771,21288,22502,22827,23605,25592,26572,26998,27012,27126,27457,27749,28282,28683,29250,29326,30109"

cd eval_scripts

poetry run python eval_scripts/ner_evaluator.py \
       -g $DIR_GOLD \
       -p $DIR_PRED \
       -d $DOCIDS \
       -l $LABELMAP \
       --rename_sentence_id \
       --label_conversion_policy DUPLICATE
