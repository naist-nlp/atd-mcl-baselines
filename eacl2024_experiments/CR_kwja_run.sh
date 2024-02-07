#!/bin/bash

## Need to set appropriate paths

DIR_GOLD=../../atd-mcl/atd-mcl/full/main/mention_tsv_per_doc/set-b
DOR_GOLD_TXT=../../eacl2024_experiments/gold/txt
DIR_PRED_KNP0=../../eacl2024_experiments/results/MR/kwja/knp
DIR_PRED_KNP=../../eacl2024_experiments/results/MR/kwja/knp_mod
DIR_PRED_JSON=../../eacl2024_experiments/results/CR/kwja/json
DOCIDS="00036,00265,00351,01059,01106,01563,01573,01575,01581,01591,02012,02053,02274,02819,03285,03417,03697,04416,04919,05412,06025,06984,07149,07205,07279,07528,07530,07559,07725,07860,07883,08966,09033,09034,09049,09139,09591,09673,09725,09726,09996,10575,10727,11421,11773,12430,12615,12785,13190,13706,14084,15003,15009,15488,17585,17640,17728,18249,19078,19531,20688,20691,20697,20771,21288,22502,22827,23605,25592,26572,26998,27012,27126,27457,27749,28282,28683,29250,29326,30109"

cd data_preprocessor/kwja_tools

mkdir -p $DOR_GOLD_TXT
mkdir -p $DIR_PRED_KNP0
mkdir -p $DIR_PRED_KNP
mkdir -p $DIR_PRED_JSON

for FILE_GOLD in `ls $DIR_GOLD/*.tsv`; do
    NAME=`basename $FILE_GOLD`
    DOCID=${NAME%.*}

    if [[ ! "$DOCIDS" =~ "$DOCID" ]]; then
        continue
    fi

    FILE_GOLD_TXT=$DOR_GOLD_TXT/$DOCID.txt
    FILE_PRED_KNP0=$DIR_PRED_KNP0/$DOCID.knp
    FILE_PRED_KNP=$DIR_PRED_KNP/$DOCID.knp

    ################
    ## The following steps are not necessary if they were already done using MR_kwja_run.sh

    # if [ "$DOCID" = "23605" ]; then
    #     # Remove the control character in the original text
    #     cut -f3 $FILE_GOLD | sed -e 's/　//g'> $FILE_GOLD_TXT
    # else
    #     cut -f3 $FILE_GOLD > $FILE_GOLD_TXT
    # fi

    # echo Input: $FILE_GOLD_TXT
    # kwja --device gpu --filename $FILE_GOLD_TXT > $FILE_PERD_KNP0
    # echo Output: $FILE_PRED_KNP0

    # ## To avoid error in pyknp/knp/syngraph.py
    # sed -e 's/^!/！/g' -e 's/^* ＊/＊ ＊/g' -e 's/^+ ＋/＋ ＋/g' $FILE_PRED_KNP0 \
    #     > $FILE_PRED_KNP
    ################

    FILE_PRED_JSON=$DIR_PRED_JSON/$DOCID.json
    poetry run python kwja_tools/convert_kwja_knp.py \
           -m coref \
           -t $FILE_GOLD_TXT \
           -k $FILE_PRED_KNP \
           -o $FILE_PRED_JSON
done
