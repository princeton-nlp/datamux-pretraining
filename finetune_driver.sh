WANDB_DISABLED=0
USE_SLURM=0
MUXING="gaussian_hadamard"
DEMUXING="index_pos"
# NUM_SENTENCES_LIST=(2 5 10)
NUM_SENTENCES_LIST=(2)
# TASK_NAMES=("qqp" "mnli" "sst2" "qnli")
# MODEL_TYPE="electra"
MODEL_TYPE="bert"
# TASK_NAMES=("pos" "qnli" "ner")
TASK_NAMES=("qnli")
# CONFIG_TYPES=("base" "large" "small")
CONFIG_TYPES=("base")
for CONFIG_TYPE in ${CONFIG_TYPES[@]}; do
    for NUM_SENTENCES in ${NUM_SENTENCES_LIST[@]}; do
        for TASK_NAME in ${TASK_NAMES[@]}; do
            TRAIN_TYPE="finetuning"
            MODEL_PATH="princeton-nlp/mux${MODEL_TYPE}_${CONFIG_TYPE}_${MUXING}_${DEMUXING}_${NUM_SENTENCES}"
            if [ $NUM_SENTENCES -eq 1 ]; then
                TRAIN_TYPE="baseline"
                MODEL_PATH="princeton-nlp/${MODEL_TYPE}_${CONFIG_TYPE}_1"
            fi
            RUN_SCRIPT="run_glue.sh"
            if [ "$TASK_NAME" = "ner" ] || [ "$TASK_NAME" = "pos" ]; then
                RUN_SCRIPT="run_ner.sh"
            fi
            CMD="sh ${RUN_SCRIPT} \
            -N $NUM_SENTENCES \
            -d ${DEMUXING} \
            -m ${MUXING} \
            -s ${TRAIN_TYPE} \
            --config_name "datamux_pretraining/configs/${MODEL_TYPE}_${CONFIG_TYPE}.json" \
            --lr 5e-5 \
            --gradient_accumulation 4 \
            --task $TASK_NAME \
            --model_path $MODEL_PATH \
            --model_type $MODEL_TYPE \
            --do_eval \
            --do_train"
            if [ $USE_SLURM -eq 1 ]; then
                sbatch -A allcs --time=12:00:00 --mem=32G --output=logs/%x-%j.out --job-name=${TASK_NAME}_${MODEL_TYPE}_${NUM_SENTENCES}_${CONFIG_TYPE} --gres=gpu:a5000:1 ./run_job.sh "$CMD"
            else
                ./run_job.sh "$CMD"
            fi
        done
    done
done 