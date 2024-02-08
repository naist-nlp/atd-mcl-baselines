################
#!/bin/bash

## Need to set appropriate paths

DIR_GOLD_DB=../../eacl2024_experiments/results/MR/spacy/gold_docbin
MODEL_DIR=../../eacl2024_experiments/results/MR/spacy/model_atd-mcl_split-118
CONF=../../eacl2024_experiments/results/MR/spacy/conf/config.cfg

cd ent_tools/ent_tools_spacy

mkdir -p $MODEL_DIR

poetry run python -m spacy train $CONF \
       --output $MODEL_DIR \
       --paths.train $DIR_GOLD_DB/train.spacy \
       --paths.dev $DIR_GOLD_DB/dev.spacy

################################################################
## NOTE:
## - During executing the above `python -m spacy` command, The following error may occur.
## 
##   File "/[path]/atd-mcl-baselines/ent_tools/ent_tools_spacy/.venv/lib/python3.10/site-packages/transformers/models/auto/tokenization_auto.py", line 638, in from_pretrained
##     raise ValueError(
## ValueError: Tokenizer class sudachitra.tokenization_electra_sudachipy.ElectraSudachipyTokenizer does not exist or is not currently imported.
##
## - Then, executing MR_spacy_path.sh may resolve the error.
