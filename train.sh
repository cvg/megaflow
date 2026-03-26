#!/bin/bash
#SBATCH --job-name=megaflow
#SBATCH --ntasks-per-node=4   
#SBATCH --gpus-per-node=4   
#SBATCH --nodes=16

# MegaFlow was trained on **64 NVIDIA GH200 GPUs across 16 nodes** using SLURM.

# --- Multi-stage training curriculum ---
# Stage 1: FlyingChairs (2-frame, no temporal attention)
python -m scripts.train_mf --cfg config/train/Chairs.json

# Stage 2: + TartanAir (2-frame)
python -m scripts.train_mf --cfg config/train/C-Tartan.json

# Stage 3: + FlyingThings3D (2-frame)
python -m scripts.train_mf --cfg config/train/C-Tartan-T-2f.json

# Stage 4: FlyingThings3D Multi-frame
python -m scripts.train_mf --cfg config/train/C-Tartan-T-mf.json

# Stage 5: + Things/Sintel/KITTI/HD1K (TSKH) multi-frame
python -m scripts.train_mf --cfg config/train/C-Tartan-T-TSKH-mf.json

# Stage 6 (Optional): Point tracking on Kubric
# python -m scripts.train_track --cfg config/train/Kubric.json
