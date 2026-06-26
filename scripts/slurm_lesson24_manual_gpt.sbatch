#!/bin/bash
#SBATCH --job-name=lesson24_gpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --qos=standard
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "============================================================"
echo "Lesson 24: Manual Hugging Face Causal LM Fine-tuning"
echo "============================================================"
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "User: $USER"
echo "Working directory: $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}"

echo ""
echo "nvidia-smi at start:"
nvidia-smi

echo ""
echo "Activating environment..."
source ./activate_vision_env.sh

echo ""
echo "Python package check:"
python - << 'EOF'
import torch
import transformers
import datasets

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("gpu memory allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
EOF

echo ""
echo "Running Lesson 24..."
python lesson24_manual_hf_causal_lm/01_manual_finetune_distilgpt2.py

echo ""
echo "nvidia-smi at end:"
nvidia-smi

echo ""
echo "Job finished at: $(date)"
