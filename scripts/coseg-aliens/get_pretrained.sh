#!/usr/bin/env bash

CHECKPOINT_DIR=$(dirname $0)/'../../checkpoints'

mkdir -p $CHECKPOINT_DIR && cd $CHECKPOINT_DIR
wget --content-disposition https://cloud.tsinghua.edu.cn/f/60ef71b57a514e4a8dbb/?dl=1
echo "downloaded the checkpoint and putting it in: " $CHECKPOINT_DIR
