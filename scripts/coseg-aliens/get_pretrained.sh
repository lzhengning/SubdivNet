#!/usr/bin/env bash

CHECKPOINT_DIR=$(dirname $0)/'../../checkpoints'

mkdir -p $CHECKPOINT_DIR && cd $CHECKPOINT_DIR
wget --content-disposition https://cloud.tsinghua.edu.cn/f/ea906194f64e40a09296/?dl=1
echo "downloaded the checkpoint and putting it in: " $CHECKPOINT_DIR
