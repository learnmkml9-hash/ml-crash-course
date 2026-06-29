#!/bin/bash
#SBATCH --job-name=lesson26_real_dataset_causal_lm
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
echo "Lesson 26: Real-Dataset Causal LM Fine-tuning"
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

export HF_HOME=/scratch/user/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/user/$USER/hf_cache/datasets
export TRANSFORMERS_CACHE=/scratch/user/$USER/hf_cache/transformers
export TOKENIZERS_PARALLELISM=false

OUTPUT_DIR=/scratch/user/$USER/ml-runs/lesson26_real_dataset_causal_lm/${SLURM_JOB_ID}
mkdir -p "$OUTPUT_DIR"

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
echo "Running Lesson 26 real-dataset causal LM experiment..."
python src/train_causal_lm.py \
    --dataset_name roneneldan/TinyStories \
    --model_checkpoint distilgpt2 \
    --block_size 128 \
    --batch_size 8 \
    --num_epochs 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --seed 0 \
    --output_dir "$OUTPUT_DIR" \
    "$@"

echo ""
echo "nvidia-smi at end:"
nvidia-smi

echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Job finished at: $(date)"
