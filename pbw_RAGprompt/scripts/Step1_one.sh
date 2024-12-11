#!/bin/bash
export PYTHONPATH="./:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7
python  pbw_RAGprompt/Step1_gen_attribute.py \
        --config pbw_RAGprompt/config/rag_gen_attribute.yaml\
        --batchsize 1
wait

## bash pbw_RAGprompt/scripts/attribute_one.sh