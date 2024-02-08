#!/bin/bash

# Need to be set
PY_VERSON=

TFM_CODE=ent_tools/ent_tools_spacy/.venv/lib/python${PY_VERSON}/site-packages/transformers/models/auto/tokenization_auto.py
TFM_CODE_SNP=eacl2024_experiments/results/MR/spacy/code/tokenization_auto_snippet.txt

cp $TFM_CODE $TFM_CODE.bak
NUM_LINES=`wc -l $TFM_CODE | cut -d' ' -f1`
NUM_LINES_INS=`grep -n "        return getattr(main_module, class_name)" $TFM_CODE | cut -d':' -f1`
NUM_LINES_REM=$(($NUM_LINES-$NUM_LINES_INS))
cat <(head -$NUM_LINES_INS $TFM_CODE.bak) $TFM_CODE_SNP <(tail -n $NUM_LINES_REM $TFM_CODE.bak) > $TFM_CODE

echo \"$TFM_CODE\" has been changed:
diff $TFM_CODE $TFM_CODE.bak

################################################################
## NOTE: You can revert the changed file with the following command.
##
## mv $TFM_CODE.bak $TFM_CODE 
