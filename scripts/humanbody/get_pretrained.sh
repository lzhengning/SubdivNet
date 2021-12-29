#!/usr/bin/env bash

CHECKPOINT_DIR=$(dirname $0)/'../../checkpoints'

mkdir -p $CHECKPOINT_DIR && cd $CHECKPOINT_DIR
wget --content-disposition https://cloud.tsinghua.edu.cn/f/9ea515945bb2452696e8/?dl=1
echo "downloaded the checkpoint and putting it in: " $CHECKPOINT_DIR
