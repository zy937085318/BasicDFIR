#!/usr/bin/env bash

NCCL_SHM_DISABLE=1 nohup python -m torch.distributed.run \
--nproc_per_node=4 basicsr/train.py -opt options/train/SwinIR/train_SwinIR_SRx2_scratch.yml\
 --launcher pytorch>>swinx2_4gpu_b8.nohup & .