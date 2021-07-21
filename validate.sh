#!/bin/bash

CHECKPOINT_PATH="/l/pytorch-image-models/output/train/imagenet_original/checkpoint-171.pth.tar"

INVERTED_CHECKPOINT_PATH="/l/pytorch-image-models/output/train/imagenet_inverted/checkpoint-127.pth.tar"

MODEL_NAME="mobilenetv3_large_100"
VAL_PATH="/l/imagenet/imagenet_files/" #validation"

python validate.py $VAL_PATH --model $MODEL_NAME --checkpoint $INVERTED_CHECKPOINT_PATH -j 12 --native-amp -b 128 --invert-images --no-prefetcher