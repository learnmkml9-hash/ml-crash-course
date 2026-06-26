#!/bin/bash
#SBATCH --job-name=gpu_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --qos=standard
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "User: $USER"
echo "Working directory: $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}"

echo ""
echo "nvidia-smi:"
nvidia-smi

echo ""
echo "Activating environment..."
source ./activate_vision_env.sh

echo ""
echo "Python and PyTorch GPU test:"
python - << 'EOF'
import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(0))

    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    print("matrix multiply OK")
    print("y shape:", y.shape)
    print("y device:", y.device)
else:
    raise RuntimeError("CUDA is not available inside the Slurm job.")
EOF

echo ""
echo "Job finished at: $(date)"
