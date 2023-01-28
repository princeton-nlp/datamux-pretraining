#!/bin/bash

DEBUG=1

if [[ $DEBUG = 1 ]]; then
    export WANDB_ENTITY="murahari"
    export WANDB_PROJECT=pretraining_insights
else
    export WANDB_ENTITY="princeton-nlp"
    export WANDB_PROJECT=datamux-pretraining
fi

NUM_INSTANCES=5

RETRIEVAL_LOSS_COEFF=1
TASK_LOSS_COEFF=0
MUXING="gaussian_hadamard"
DEMUXING="index_pos"
LEARNING_RATE=1e-4

OUTPUT_DIR=${OUTPUT_DIR}_${LOSS_TYPE}_${VERSION}_${LEARNING_RATE}
VERSION="bert"
RUN_NAME=bert_${NUM_INSTANCES}_${MUXING}_${DEMUXING}_${LEARNING_RATE}_${TASK_LOSS_COEFF}_${RETRIEVAL_LOSS_COEFF}
OUTPUT_DIR="checkpoints/bert_${NUM_INSTANCES}_${MUXING}_${DEMUXING}_${LEARNING_RATE}_${TASK_LOSS_COEFF}_${RETRIEVAL_LOSS_COEFF}"

if [[ $DEBUG = 1 ]]; then
    CMD="python datamux_pretraining/run_pretraining.py"
else
    CMD="python -m torch.distributed.launch --nproc_per_node=8 datamux_pretraining/run_pretraining.py"
fi
CMD_ARGS="--config_name datamux_pretraining/configs/bert_base.json \
--tokenizer_name bert-base-uncased \
--dataset_name bookcorpus \
--do_train \
--do_eval \
--output_dir $OUTPUT_DIR \
--max_seq_length 512 \
--per_device_train_batch_size $((16 * NUM_INSTANCES)) \
--per_device_eval_batch_size $((32 * NUM_INSTANCES)) \
--learning_rate $LEARNING_RATE \
--max_steps 1000000 \
--lr_scheduler_type linear \
--warmup_steps 10000 \
--run_name $RUN_NAME \
--logging_steps 100 \
--save_steps 10000 \
--overwrite_cache 0 \
--eval_steps 10000 \
--evaluation_strategy steps \
--num_instances $NUM_INSTANCES \
--mlm_probability 0.15 \
--report_to wandb \
--model_version $VERSION \
--dataloader_drop_last 1 \
--muxing_variant ${MUXING} \
--demuxing_variant ${DEMUXING} \
--retrieval_loss_coeff ${RETRIEVAL_LOSS_COEFF} \
--task_loss_coeff ${TASK_LOSS_COEFF} \
--save_total_limit 4 \
--load_best_model_at_end 1 \
--metric_for_best_model eval_loss \
--dataloader_num_workers 4 \
--preprocessing_num_workers 1 \
--validation_split_percentage 5 \
--fp16 1"
CMD="$CMD $CMD_ARGS"
if [[ $DEBUG = 1 ]]; then
    ./run_job.sh "$CMD"
else
    sbatch -A pnlp --time=$TIME --mem=32G --output=logs/%x-%j.out --job-name=${TASK_NAME}_${NUM_SENTENCES}_${INFERENCE_TYPE}_${LOSS_TYPE} --gres=gpu:a6000:8 ./run_job.sh \
    "$CMD"
fi