#!/usr/bin/env bash

DATADIR=$(dirname $0)/'../../data'

mkdir -p $DATADIR && cd $DATADIR
wget --content-disposition https://cloud.tsinghua.edu.cn/f/27cc19807ff644989881/?dl=1
echo "downloaded the data and putting it in: " $DATADIR
echo "unzipping"
unzip -q coseg-aliens-MAPS-256-3.zip && rm coseg-aliens-MAPS-256-3.zip
