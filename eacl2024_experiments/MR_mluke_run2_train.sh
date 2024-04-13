################
#!/bin/bash

## Need to set appropriate paths

DIR_GOLD=../eacl2024_experiments/results/MR/mluke/gold
MODEL_DIR=../eacl2024_experiments/results/MR/mluke/model_atd-mcl_split-118

cd mr_mluke

mkdir -p $MODEL_DIR

poetry run torchrun --nproc_per_node 4 luke-ner/src/main.py \
       --do_train \
       --do_eval \
       --do_predict \
       --train_file $DIR_GOLD/train.jsonl \
       --validation_file $DIR_GOLD/dev.jsonl \
       --test_file $DIR_GOLD/test.jsonl \
       --model "studio-ousia/mluke-large-lite" \
       --output_dir $MODEL_DIR \
       --per_device_train_batch_size 2 \
       --per_device_eval_batch_size 8 \
       --max_entity_length 64 \
       --max_mention_length 16 \
       --pretokenize \
       --num_train_epochs 20 \
       --learning_rate 1e-5 \
       --save_strategy epoch \
       --load_best_model_at_end \
       --metric_for_best_model eval_f1
