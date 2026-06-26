#!/bin/bash

export MY_SCRATCH=/scratch/user/$USER

source $MY_SCRATCH/ml-envs/ml-crash-venv/bin/activate

export PIP_CACHE_DIR=$MY_SCRATCH/pip_cache

export HF_HOME=$MY_SCRATCH/hf_cache
export HF_DATASETS_CACHE=$MY_SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$MY_SCRATCH/hf_cache/transformers

export TOKENIZERS_PARALLELISM=false

echo "Activated VISION ML environment"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "MY_SCRATCH: $MY_SCRATCH"
echo "HF_HOME: $HF_HOME"
