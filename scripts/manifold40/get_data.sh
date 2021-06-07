#!/usr/bin/env bash

DATADIR=$(dirname $0)/'../../data'

mkdir -p $DATADIR && cd $DATADIR
wget --content-disposition https://cloud.tsinghua.edu.cn/f/2a292c598af94265a0b8/?dl=1
echo "downloaded the data and putting it in: " $DATADIR
echo "unzipping"
unzip -q Manifold40.zip && rm Manifold40.zip
