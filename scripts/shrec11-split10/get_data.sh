#!/usr/bin/env bash

DATADIR=$(dirname $0)/'../../data'

mkdir -p $DATADIR && cd $DATADIR
wget --content-disposition https://cloud.tsinghua.edu.cn/f/6e59181fd26d4a2dbad9/?dl=1
echo "downloaded the data and putting it in: " $DATADIR
echo "unzipping"
unzip -q SHREC11-MAPS-48-4-split10.zip && rm SHREC11-MAPS-48-4-split10.zip
