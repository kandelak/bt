#!/bin/bash

#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -o enroot_test.out
#SBATCH -e enroot_test.err

srun  --container-image='/dss/dsshome1/05/ge54not2/images/sam-adapter-ready.sqsh' \
      --container-workdir='/workspace/PV-Segmentation_SAM_Adapter' \
      torchrun train_cuda_logmemory.py --config configs/ma_B_cuda.yaml

