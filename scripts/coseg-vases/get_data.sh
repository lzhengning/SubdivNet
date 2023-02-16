#!/usr/bin/env bash

DATADIR=$(dirname $0)/'../../data'

mkdir -p $DATADIR && cd $DATADIR
wget --content-disposition https://cg.cs.tsinghua.edu.cn/dataset/subdivnet/datasets/coseg-vases-MAPS-256-3.zip
echo "downloaded the data and putting it in: " $DATADIR
echo "unzipping"
unzip -q coseg-vases-MAPS-256-3.zip && rm coseg-vases-MAPS-256-3.zip
