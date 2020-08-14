#!/usr/bin/env bash
#source activate remind_proj

lr=2e-3
CONFIG=TDIUC_streaming
export PYTHONPATH=/hdd/robik/projects/REMIND

#DATA_ORDER=iid
#expt=${CONFIG}_${DATA_ORDER}_${lr}

#CUDA_VISIBLE_DEVICES=0 nohup python -u vqa_experiments/vqa_trainer.py \
#--config_name ${CONFIG} \
#--expt_name ${expt} \
#--stream_with_rehearsal \
#--data_order ${DATA_ORDER} \
#--lr ${lr} &> logs/${expt}.log &

DATA_ORDER=qtype # or qtype
expt=${CONFIG}_${DATA_ORDER}_${lr}

CUDA_VISIBLE_DEVICES=0 python -u vqa_experiments/vqa_trainer.py \
--config_name ${CONFIG} \
--expt_name ${expt} \
--stream_with_rehearsal \
--data_order ${DATA_ORDER} \
--lr ${lr}
