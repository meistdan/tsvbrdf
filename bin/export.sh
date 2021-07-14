#!/bin/bash

TSVBRDF=tsvbrdf.exe

TESTS='
AgedChrome
DuctTape
FractureAsphalt
Grass
Metal
SnowyGround
GoldFlake
OldWood
StoneDirt
WoodLog
tvBTF09
tvBTF20
tvBTF22
tvBTF23
tvBTF24
tvBTF25
tvBTF27
tvBTF29
tvBTF30
tvBTF31
tvBTF32
tvBTF35
tvBTF37
tvBTF39
tvBTF40
tvBTF41
tvBTF42
tvBTF43
tvBTF44
tvBTF45
'

TESTS='
tvBTF31
'

#TESTS='WoodLog'

DATA_TYPE=poly-5

DATA=../data
CURRENT_DIR=$PWD

START=$(date)
for TEST in $TESTS; do
	echo $TEST
	#MAT_DATA=$DATA/spatial/$TEST/$DATA_TYPE
	MAT_DATA=$DATA/original/$TEST/$DATA_TYPE
	IMAGES=$MAT_DATA/images
	mkdir -p $IMAGES
	./$TSVBRDF $MAT_DATA
	cp clip.sh clip.avs submission.vcf $IMAGES
	cd $IMAGES
	./clip.sh
	cd $CURRENT_DIR
done
END=$(date)

echo $START
echo $END
