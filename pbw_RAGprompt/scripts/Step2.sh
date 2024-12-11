#!/bin/bash
export PYTHONPATH="./:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=4,5,6,7
python  pbw_RAGprompt/Step2_summary_caption.py \
        --config pbw_RAGprompt/config/rag_summary.yaml
wait

## bash pbw_RAGprompt/scripts/Step2.sh