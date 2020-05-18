#!/bin/bash

METHOD=$1
SEED=$2
SEARCH_SPACE=$3
EPOCHS=$4
ARCH_LR=$5
EDGE_LR=$6

S3_BUCKET=$7

cd /code/nasbench-1shot1
PYTHONPATH=$PWD python optimizers/$METHOD/train_search.py \
    --seed=$SEED \
    --save=${METHOD}_space${SEARCH_SPACE}_seed${SEED} \
    --epochs=$EPOCHS \
    --search_space=${SEARCH_SPACE} \
    --s3_bucket=${S3_BUCKET} \
    --arch_learning_rate=${ARCH_LR} \
    --edge_learning_rate=${EDGE_LR} $7 $8 $9 ${10}
