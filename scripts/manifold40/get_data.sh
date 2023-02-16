#!/usr/bin/env bash

DATADIR=$(dirname $0)/'../../data'

mkdir -p $DATADIR && cd $DATADIR
wget --content-disposition https://cg.cs.tsinghua.edu.cn/dataset/subdivnet/datasets/Manifold40-MAPS-96-3.zip
echo "downloaded the data and putting it in: " $DATADIR
echo "unzipping"
unzip -q Manifold40-MAPS-96-3.zip && rm Manifold40-MAPS-96-3.zip
