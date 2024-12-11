#!/bin/bash

export PYTHONPATH="./:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python  pbw_RAGprompt/test_2bVL.py \
        --config pbw_RAGprompt/config/rag_test.yaml\
        --batchsize 1\
        --num-chunks ${CHUNKS}  \
        --chunk-idx ${IDX}  &
done
wait
## bash pbw_RAGprompt/scripts/test.sh