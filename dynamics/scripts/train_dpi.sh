#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py --model_name GNSRigidH --log_per_iter 1000 --training_fpt 1 --ckp_per_iter 5000 --outf "$DPI"
