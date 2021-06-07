#!/usr/bin/env bash

CHECKPOINT_DIR=$(dirname $0)/'../../checkpoints'

mkdir -p $CHECKPOINT_DIR && cd $CHECKPOINT_DIR
wget --content-disposition https://cloud.tsinghua.edu.cn/f/b7b733157e7d44ea9bca/?dl=1
echo "downloaded the checkpoint and putting it in: " $CHECKPOINT_DIR
