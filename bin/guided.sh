#!/bin/bash

TSVBRDF=tsvbrdf.exe

DATA_TYPE=poly-3

DATA=../data
CURRENT_DIR=$PWD

TIME=0.0
TEST='SnowyGround'
SOURCE=$DATA/original/$TEST/$DATA_TYPE
TARGET=$DATA/cag.png

echo $TEST
GUIDED_DATA=$DATA/guided/$TEST/$DATA_TYPE
GUIDED_IMAGES=$GUIDED_DATA/images
mkdir -p $GUIDED_DATA
mkdir -p $GUIDED_IMAGES
./$TSVBRDF $SOURCE $TARGET $GUIDED_DATA $TIME
#cp clip.sh clip.avs submission.vcf $GUIDED_IMAGES
#cd $GUIDED_IMAGES
#./clip.sh
#cd $CURRENT_DIR
