#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,3,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m infer \
        --config config/rag_clip_train_task2.yaml\
        --batchsize 1\
        --num-chunks ${CHUNKS}  \
        --chunk-idx ${IDX}  &
done
wait
#bash scripts/rag_train_task2.sh