#!/usr/bin/env bash

PROJ_ROOT=/home/tyler/Desktop/codes/REMIND

export PYTHONPATH=${PROJ_ROOT}
source activate base
cd ${PROJ_ROOT}/image_classification_experiments

IMAGE_DIR=/media/tyler/nvme_drive/data/ImageNet2012
EXPT_NAME=remind_imagenet
GPU=0

REPLAY_SAMPLES=50
MAX_BUFFER_SIZE=959665
CODEBOOK_SIZE=256
NUM_CODEBOOKS=32
BASE_INIT_CLASSES=100
CLASS_INCREMENT=100
NUM_CLASSES=1000

CUDA_VISIBLE_DEVICES=${GPU} python -u imagenet_experiment.py \
--images_dir ${IMAGE_DIR} \
--max_buffer_size ${MAX_BUFFER_SIZE} \
--num_classes ${NUM_CLASSES} \
--streaming_min_class ${BASE_INIT_CLASSES} \
--streaming_max_class ${NUM_CLASSES} \
--base_init_classes ${BASE_INIT_CLASSES} \
--class_increment ${CLASS_INCREMENT} \
--classifier ResNet18_StartAt_Layer4_1 \
--classifier_ckpt ./imagenet_files/best_ResNet18ClassifyAfterLayer4_1_100.pth \
--rehearsal_samples ${REPLAY_SAMPLES} \
--start_lr 0.1 \
--end_lr 0.001 \
--lr_step_size 100 \
--lr_mode step_lr_per_class \
--weight_decay 1e-5 \
--random_resized_crops \
--use_mixup \
--mixup_alpha .1 \
--label_dir ./imagenet_files/ \
--num_codebooks ${NUM_CODEBOOKS} \
--codebook_size ${CODEBOOK_SIZE} \
--expt_name ${EXPT_NAME} > logs/${EXPT_NAME}.log
