#!/usr/bin/env bash

NCCL_SHM_DISABLE=1 nohup python -m torch.distributed.run \
--nproc_per_node=4 basicsr/train.py -opt options/train/JiT/train_swinflowir_S_SRx8_scratch_B8_P32.yml\
 --launcher pytorch & .