# Fixing Malfunctional Objects With Learned Physical Simulation and Functional Prediction

![Teaser](http://fixing-malfunctional.csail.mit.edu/assets/teaser.png)

**Figure 1. Teaser.**

## Introduction

This paper studies the problem of fixing malfunctional 3D objects. While previous works focus on building passive perception models to learn the functionality from static 3D objects, we argue that functionality is reckoned with respect to the physical interactions between the object and the user. Given a malfunctional object, humans can perform mental simulations to reason about its functionality and figure out how to fix it. Inspired by this, we propose FixIt, a dataset that contains about 5k poorly-designed 3D physical objects paired with choices to fix them. To mimic humans' mental simulation process, we present FixNet, a novel framework that seamlessly incorporates perception and physical dynamics. Specifically, FixNet consists of a perception module to extract the structured representation from the 3D point cloud, a physical dynamics prediction module to simulate the results of interactions on 3D objects, and a functionality prediction module to evaluate the functionality and choose the correct fix. Experimental results show that our framework outperforms baseline models by a large margin, and can generalize well to objects with similar interaction types.

This paper is accepted by CVPR2022.

Project Page: http://fixing-malfunctional.csail.mit.edu/

## Data and Checkpoints
[Download Link](https://drive.google.com/drive/folders/1h9kMRilQcjbD4Tyt58pmMUEnMIicNATi?usp=sharing)
We are still updating the dataset and Checkpoints

## SetUp
'''
conda create -n fixing python=3.7 pytorch=1.4.0 torchvision -c pytorch
conda activate fixing

pip install -r requirements.txt

cd flownet3d/lib
python setup.py install
'''

## Flow Prediction Network
'''
cd flownet3d
python main.py --exp_name fridge #for training
python main.py --exp_name fridge --eval #for testing
cd utils
python hungarian.py 
python hungarian.py --split test
'''

## Instance Segmentation Network
'''
cd pointnet++ 
python train_insseg.py --model pointnet2_part_seg_msg --log_dir fridge
python test_insseg.py --model pointnet2_part_seg_msg --log_dir fridge
'''

## Dynamics Prediction Module
'''
# Training
cd utils
python rearrange_data_for_dpi.py
python rearrange_data_for_dpi.py --split test
python create_train_valid.py
cd ../
cd dynamics
bash scripts/train_dpi.sh
# Inference
cd utils
python rearrange_test_data_for_dpi.py
python rearrange_test_data_for_dpi.py --split test
cd dynamics
bash scripts/eval_dpi.sh
'''

## Functional Prediction Module
'''
cd pointnet++
python train_classification.py  --model pointnet2_cls_ssg --log_dir fridge
python test_classification.py fridge
cd ../
cd utils
python calculate_final_result.py
'''

@inproceedings{hong2021fixing,
 author = {Hong, Yining and Mo, Kaichun and Yi, Li and Guibas, Leonidas J and Torralba, Antonio and Tenenbaum, Joshua B and Gan, Chuang},
 title = {Fixing Malfunctional Objects With Learned Physical Simulation and Functional Prediction},
 booktitle = {CVPR},
 year = {2022}
}
