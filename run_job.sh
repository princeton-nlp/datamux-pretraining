#!/bin/bash
export WANDB_NOTES=$SLURM_JOB_ID
# trap 'echo signal recieved!; kill -s SIGUSR1 "${PID}"; wait "${PID}"' USR1
# trap 'echo signal recieved!; kill "${PID}"' SIGINT
$@
# PID="$!"
# echo $PID
# wait "${PID}"
# for var in "$@"
# do
#     echo $var
#     echo "var"
#     $var
# done