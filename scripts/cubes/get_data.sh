#!/usr/bin/env bash

DATADIR=$(dirname $0)/'../../data'

mkdir -p $DATADIR && cd $DATADIR
wget --content-disposition https://cg.cs.tsinghua.edu.cn/dataset/subdivnet/datasets/Cubes-MAPS-48-4.zip
echo "downloaded the data and putting it in: " $DATADIR
echo "unzipping"
unzip -q Cubes-MAPS-48-4.zip && rm Cubes-MAPS-48-4.zip
