#!/usr/bin/env bash

CHECKPOINT_DIR=$(dirname $0)/'../../checkpoints'

mkdir -p $CHECKPOINT_DIR && cd $CHECKPOINT_DIR
wget --content-disposition https://cloud.tsinghua.edu.cn/f/1f86eee93a1e4dd7a690/?dl=1
echo "downloaded the checkpoint and putting it in: " $CHECKPOINT_DIR
