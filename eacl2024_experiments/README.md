# Scripts for the EACL 2024 Paper Experiments

## 1. Preparation

### ATD-MCL

1. Obtain and restore the [ATD-MCL](https://github.com/naist-nlp/atdmcl) dataset.
1. Place `atd-mcl` directory under `atd-mcl-baselines` directory (or create a symbolic link).

### eval_scripts

- Install `eval_scripts`.

    ~~~~
    cd eval_scripts
    poetry install
    cd ..
    ~~~~

### Preprocessed OSM Data for Entity Disambiguation

- Decompress the preprocessed OSM data.

    ~~~~
    cd data/osm
    tar -jxvf 20230620_all_extnames.txt.tar.bz2
    ~~~~

## 2. Inter-Annotator Agreement Assessment

### 2.1 Mention Annotation

- Execute the experimental script.

    ~~~~
    ./eacl2024_experiments/MR_calc_IAA.sh
    ~~~~

### 2.2 Coreference Annotation

- Execute the experimental script.

    ~~~~
    ./eacl2024_experiments/CR_calc_IAA.sh
    ~~~~

### 2.3 Link Annotation

- Execute the experimental script.

    ~~~~
    ./eacl2024_experiments/ED_calc_IAA.sh
    ~~~~

## 3. System Performance Evaluation

### 3.1 Rule-CR1/CR2/ED

- Install `rule_based`.

    ~~~~
    cd rule_based
    poetry install
    cd ..
    ~~~~

#### Rule-CR1/CR2

- Execute the experimental script.

    ~~~~
    ./eacl2024_experiments/CR_rule_run.sh
    ./eacl2024_experiments/CR_rule_eval.sh
    ~~~~

#### Rule-ED

- Execute the experimental script.

    ~~~~
    ./eacl2024_experiments/ED_rule_run.sh
    ./eacl2024_experiments/ED_rule_eval.sh
    ~~~~

### 3.2 GiNZA

- Install GiNZA. See https://github.com/megagonlabs/ginza

- Execute the experimental script.

    ~~~~
    ./eacl2024_experiments/MR_ginza_run.sh
    ./eacl2024_experiments/MR_ginza_eval.sh
    ~~~~

### 3.3 KWJA

- Install KWJA. See https://github.com/ku-nlp/kwja

- Install `kwja_tools`.

    ~~~~
    cd data_preprocessor/kwja_tools
    poetry install
    cd ../..
    ~~~~

#### Mention Recognition

- Execute the experimental script.

    ~~~~
    ./eacl2024_experiments/MR_kwja_run.sh
    ./eacl2024_experiments/MR_kwja_eval.sh
    ~~~~

#### Coreference Resolution

- Execute the experimental script.

     
    ~~~~
    ./eacl2024_experiments/CR_kwja_run.sh
    ./eacl2024_experiments/CR_kwja_eval.sh
    ~~~~

### 3.4 spaCy-MR

- Install `ent_tools_spacy`.

    ~~~~
    git clone https://github.com/shigashiyama/ent_tools
    cd ent_tools/ent_tools_spacy
    poetry install
    cd ../..
    ~~~~

- Execute the experimental script.

    ~~~~
    ./eacl2024_experiments/MR_spacy_run1_preproc.sh
    ./eacl2024_experiments/MR_spacy_run2_train.sh
    ./eacl2024_experiments/MR_spacy_run3_decode.sh
    ./eacl2024_experiments/MR_spacy_eval.sh
    ~~~~
    - If an error occurs during executing `MR_spacy_run2_train.sh`, then `MR_spacy_patch.sh` may resolve it.
     
        ~~~~
        ./eacl2024_experiments/MR_spacy_patch.sh
        ~~~~

### 3.5 mLUKE-MR (luke-ner)

- Install `luke-ner` (https://github.com/naist-nlp/luke-ner).

    ~~~~
    git submodule update --init --recursive
    cd mr_mluke
    poetry install
    cd ..
    ~~~~

- Execute the experimental script.

    ~~~~
    ./eacl2024_experiments/MR_mluke_run1_preproc.sh
    ./eacl2024_experiments/MR_mluke_run2_train.sh
    ./eacl2024_experiments/MR_mluke_run3_postproc.sh
    ./eacl2024_experiments/MR_mluke_eval.sh
    ~~~~

### 3.6 mLUKE-CR (luke-coref)

- Install `luke-coref` (https://github.com/naist-nlp/luke-coref).

    ~~~~
    git submodule update --init --recursive
    cd mr_mluke
    poetry install
    cd ..
    ~~~~

- Execute the experimental script.

    ~~~~
    ./eacl2024_experiments/CR_mluke_run1_preproc.sh
    ./eacl2024_experiments/CR_mluke_run2_train.sh
    ./eacl2024_experiments/CR_mluke_run3_postproc.sh
    ./eacl2024_experiments/CR_mluke_eval.sh
    ~~~~

### 3.7 BERT-ED

- Execute the experimental script.

    ~~~~
    #TODO ./eacl2024_experiments/ED_bert_run.sh
    ./eacl2024_experiments/ED_bert_eval.sh
    ~~~~
