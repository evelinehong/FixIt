#!/bin/bash
CUDA_VISIBLE_DEVICES=0  python eval.py --subsample 1000 --epoch 0 --iter 10000 --model_name GNSRigidH --training_fpt 1 --mode "test" --test_training_data_processing 1 --dataf "./output"


