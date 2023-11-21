#!/bin/bash

#SBATCH -p lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH -o enroot_test.out
#SBATCH -e enroot_test.err

srun --container-mounts=/dss/dsshome1/05/ge54not2/git/PV-Segmentation_SAM_Adapter:/app/ \
    --container-image='docker://aleksandelaki/sam-adapter-lrz' \
    python3 -m torch.distributed.launch train_cuda.py --config configs/ma_B_cuda.yaml

