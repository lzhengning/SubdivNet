#!/usr/bin/env bash

DATADIR=$(dirname $0)/'../../data'

mkdir -p $DATADIR && cd $DATADIR
wget --content-disposition https://cloud.tsinghua.edu.cn/f/f1d091376e9e4bd69ce1/?dl=1
echo "downloaded the data and putting it in: " $DATADIR
echo "unzipping"
unzip -q Cubes-MAPS-48-4.zip && rm Cubes-MAPS-48-4.zip
