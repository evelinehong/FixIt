#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import h5py
import numpy as np
import open3d as o3d
import json
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
import numpy as np
import torch

class SceneflowDataset(Dataset):
    def __init__(self, npoints=2048, cat='fridge', partition='train', sample = False):
        self.npoints = npoints
        self.partition = partition

        self.root = os.path.join("../data", cat, "shapes")

        self.split = partition
        if partition == 'val':
            self.split = 'train'
        self.datapath = os.path.join(self.root, '%s_before'%self.split)
        shapes = os.listdir(self.datapath)
    
        self.filepath = []
        self.nums = []

        for shape in shapes:
            try:
                os.mkdir(os.path.join(self.datapath, shape, "flow_pred"))
            except:
                pass
            files = os.listdir(os.path.join(self.datapath, shape, "new"))
            for file in files:
                if file.endswith(".npy") and not file.endswith("9.npy"):
                    self.filepath.append(os.path.join(self.datapath, shape, "new", file))
                    self.nums.append(int(file.replace(".npy", "")))

        if self.partition == 'train' and sample:
            self.filepath = self.filepath[:int(len(self.filepath)/5)]
            self.nums = self.nums[:int(len(self.nums)/5)]
        if self.partition == 'val' and sample:
            self.filepath = self.filepath[int(len(self.filepath)/5):int(len(self.filepath)/5*2)]
            self.nums = self.nums[int(len(self.nums)/5):int(len(self.nums)/5*2)]

    def __getitem__(self, index):
        num = self.nums[index]
        pos1 = np.load(open(self.filepath[index], "rb"), allow_pickle=True)[16:]
        pos2 = np.load(open(self.filepath[index].replace(str(num)+".npy", str(num+1)+".npy"), "rb"), allow_pickle=True)[16:]

        label = np.load(open(self.filepath[index].replace("new", "flow"), "rb"), allow_pickle=True)[16:]

        color1 = np.array([[0.0, 0.0, 0.0] for i in range(len(pos1))])
        color2 = color1

        mask1 = np.array([1 for  i in range(len(pos1))])

        return pos1, pos2, color1, color2, label, mask1, self.filepath[index]

    def __len__(self):
        return len(self.filepath)