#!/usr/bin/env bash

DATADIR=$(dirname $0)/'../../data'

mkdir -p $DATADIR && cd $DATADIR
wget --content-disposition https://cloud.tsinghua.edu.cn/f/af5c682587cc4f9da9b8/?dl=1
echo "downloaded the data and putting it in: " $DATADIR
echo "unzipping"
unzip -q Manifold40-MAPS-96-3.zip && rm Manifold40-MAPS-96-3.zip
