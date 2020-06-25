#!/usr/bin/env bash

PROJ_ROOT=/home/tyler/Desktop/codes/REMIND

export PYTHONPATH=${PROJ_ROOT}
source activate base
cd ${PROJ_ROOT}/image_classification_experiments

IMAGENET_DIR=/media/tyler/nvme_drive/data/ImageNet2012
BASE_MAX_CLASS=100
MODEL=ResNet18ClassifyAfterLayer4_1
GPU=0

CUDA_VISIBLE_DEVICES=${GPU} python -u train_base_init_network_from_scratch.py \
--arch ${MODEL} \
--data ${IMAGENET_DIR} \
--base_max_class ${BASE_MAX_CLASS} \
--labels_dir ./imagenet_files/ \
--ckpt_file ${MODEL}_${BASE_MAX_CLASS}.pth > logs/${MODEL}_${BASE_MAX_CLASS}_from_scratch.log
