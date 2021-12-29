#!/usr/bin/env bash

DATADIR=$(dirname $0)/'../../data'

mkdir -p $DATADIR && cd $DATADIR
wget --content-disposition https://cloud.tsinghua.edu.cn/f/46f7fd946cd045ee83f5/?dl=1
echo "downloaded the data and putting it in: " $DATADIR
echo "unzipping"
unzip -q coseg-vases-MAPS-256-3.zip && rm coseg-vases-MAPS-256-3.zip
