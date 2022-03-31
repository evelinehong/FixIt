import os
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser(description='Point Cloud Registration')
parser.add_argument('--category', type=str, default='fridge', metavar='N',
                    help='Name of the category')
parser.add_argument('--split', type=str, default='train', metavar='N',
                    choices=['train', 'test'],
                    help='train or test')
args = parser.parse_args()

root_dir = os.path.join("../data/%s/shapes"%args.category, "%s_before"%args.split)

for shape in os.listdir(root_dir):
    if "txt" in shape: continue
    segmented_dict = [[], [], [], [], [j+2048 for j in range(16)]]

    instance_idx = []
    instance_idx.append(0)
    clusters = []
    root_num = []
    root_des_radius = []
    spacing = 0.05
    dt = 0.01

    seg_file = np.load(open(os.path.join(root_dir, shape, "instance_segmentation.npy"), "rb"))

    for (j,label) in enumerate(seg_file):
        segmented_dict[label].append(j)

    n_objects = 0
    count = 0
    material = []

    find_fluid = False

    hand = np.load(open(os.path.join(root_dir, shape, "new", "%d.npy"%0), "rb"))[:16]

    pts = np.load(open(os.path.join(root_dir, shape, "corresponding_points", "%d.npy"%0), "rb"))
    pts = np.concatenate((pts, hand))

    for (s,seg) in enumerate(segmented_dict):
        if len(seg):
            # fluid = False
            # if len(seg) < 650:
            #     fluid = True
            # # for j in seg:
            # #     if (pts[j][0] ** 2 + pts[j][2] ** 2 + pts[j][1] ** 2) < 0.005:
            # #         find_fluid = True
            # #         fluid = True
            # # if fluid:
            # # material.append('fluid')
            # # else:
            material.append('rigid')
            count += len(seg)
            n_objects += 1
            instance_idx.append(count)
            clusters.append([[np.array([0]* len(seg), dtype=np.int32)]])
            root_num.append([1])
            root_des_radius.append([spacing])

    instance_idx.pop()
        
    try:
        os.mkdir(os.path.join(root_dir, shape, "arranged_points"))
    except:
        pass

    for i in range(10):
        pts = np.load(open(os.path.join(root_dir, shape, "corresponding_points", "%d.npy"%i), "rb"))
        pts = np.concatenate((pts, hand))

        pts_new = []
        for seg in segmented_dict:
            for s in seg:
                pts_new.append(pts[s])
        pts_new = np.array(pts_new)

        with open(os.path.join(root_dir, shape, "arranged_points", "%d.npy"%i), "wb") as f:
            np.save(f, pts_new)

    phases_dict = dict()
    phases_dict["instance_idx"] = instance_idx
    phases_dict["root_des_radius"] = root_des_radius
    phases_dict["root_num"] = root_num
    phases_dict["clusters"] = clusters
    phases_dict["time_step"] = 10
    phases_dict["n_objects"] = n_objects
    phases_dict["n_particles"] = 2048
    phases_dict["material"] = material
    phases_dict["dt"] = dt

    with open(os.path.join(root_dir, shape, "phases_dict.pkl"), "wb") as f:
        pickle.dump(phases_dict, f)