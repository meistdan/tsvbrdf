#!/bin/bash

TSVBRDF=tsvbrdf.exe

TESTS='
SnowyGround
'

DATA_TYPE=poly-3

DATA=../data
CURRENT_DIR=$PWD

START=$(date)
for TEST in $TESTS; do
	echo $TEST
	ORIGINAL_DATA=$DATA/original/$TEST/$DATA_TYPE
	ORIGINAL_IMAGES=$ORIGINAL_DATA/images
	ENLARGED_DATA=$DATA/enlarged/$TEST/$DATA_TYPE
	ENLARGED_IMAGES=$ENLARGED_DATA/images
	mkdir -p $ORIGINAL_DATA
	mkdir -p $ORIGINAL_IMAGES
	mkdir -p $ENLARGED_DATA
	mkdir -p $ENLARGED_IMAGES
	./$TSVBRDF $ORIGINAL_DATA $ENLARGED_DATA
#	cp clip.sh clip.avs submission.vcf $ORIGINAL_IMAGES
#	cd $ORIGINAL_IMAGES
#	./clip.sh
#	cd $CURRENT_DIR
#	cp clip.sh clip.avs submission.vcf $ENLARGED_IMAGES
#	cd $ENLARGED_IMAGES
#	./clip.sh
#	cd $CURRENT_DIR
done
END=$(date)

echo $START
echo $END
