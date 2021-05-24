#!/usr/bin/env bash

ARCH="UNAS_CIFAR10"
export EXPR_ID=${ARCH}
export DATA_DIR="/home/grads/y/yiwei_chen/cifar10"
export CHECKPOINT_DIR="/home/grads/y/yiwei_chen/unas/experiments"

## Evaluate discovered cell on CIFAR10 by training the cell from scratch ###
cd ./darts
python -m torch.distributed.launch --nproc_per_node=1  --master_port=8000 train_eval_cifar.py \
--data $DATA_DIR --root_dir $CHECKPOINT_DIR --save $EXPR_ID \
--genotype ${ARCH} --dataset cifar10  \
--batch_size 128 --warmup_epochs 30 --learning_rate 0.05 --drop_path_prob 0.2 \
--gpu 2
