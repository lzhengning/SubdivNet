#!/usr/bin/env bash

DATADIR=$(dirname $0)/'../../data'

mkdir -p $DATADIR && cd $DATADIR
wget --content-disposition https://cloud.tsinghua.edu.cn/f/305c270079724f689418/?dl=1
echo "downloaded the data and putting it in: " $DATADIR
echo "unzipping"
unzip -q HumanBody-NS-256-3.zip && rm HumanBody-NS-256-3.zip
