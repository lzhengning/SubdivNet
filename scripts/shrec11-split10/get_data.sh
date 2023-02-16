#!/usr/bin/env bash

DATADIR=$(dirname $0)/'../../data'

mkdir -p $DATADIR && cd $DATADIR
wget --content-disposition https://cg.cs.tsinghua.edu.cn/dataset/subdivnet/datasets/SHREC11-MAPS-48-4-split10.zip
echo "downloaded the data and putting it in: " $DATADIR
echo "unzipping"
unzip -q SHREC11-MAPS-48-4-split10.zip && rm SHREC11-MAPS-48-4-split10.zip
