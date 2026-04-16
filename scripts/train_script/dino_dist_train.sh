#!/usr/bin/env bash

NCCL_SHM_DISABLE=1 nohup python -m torch.distributed.run \
--nproc_per_node=4 basicsr/train.py -opt options/train/DINOv3/train_DinoATDUNet_SRModel_B32P64_SRx4_scratch.yml & .