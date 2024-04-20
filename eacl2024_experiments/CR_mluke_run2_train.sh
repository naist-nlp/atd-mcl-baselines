################
#!/bin/bash

## Need to set appropriate paths

DIR_GOLD=../eacl2024_experiments/results/CR/mluke/gold
MODEL_DIR=../eacl2024_experiments/results/CR/mluke/model_atd-mcl_split-118

cd cr_mluke

mkdir -p $MODEL_DIR

poetry run torchrun --nproc_per_node 4 luke-coref/src/main.py \
       --do_train \
       --do_eval \
       --do_predict \
       --train_file $DIR_GOLD/train.jsonl \
       --validation_file $DIR_GOLD/dev.jsonl \
       --test_file $DIR_GOLD/test.jsonl \
       --model "studio-ousia/mluke-large" \
       --output_dir $MODEL_DIR \
       --per_device_train_batch_size 1 \
       --per_device_eval_batch_size 4 \
       --num_train_epochs 20 \
       --learning_rate 5e-5 \
       --save_strategy epoch \
       --save_total_limit 3 \
       --load_best_model_at_end \
       --metric_for_best_model eval_accuracy
