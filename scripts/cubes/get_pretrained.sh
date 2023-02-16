#!/usr/bin/env bash

CHECKPOINT_DIR=$(dirname $0)/'../../checkpoints'

mkdir -p $CHECKPOINT_DIR && cd $CHECKPOINT_DIR
wget --content-disposition https://cg.cs.tsinghua.edu.cn/dataset/subdivnet/checkpoints/Cubes/cubes.pkl
echo "downloaded the checkpoint and putting it in: " $CHECKPOINT_DIR
