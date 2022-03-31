import os
import numpy as np
import json
import pickle
import math
import copy
import argparse

def Rz(theta):
    return np.matrix([[ math.cos(theta), -math.sin(theta), 0 ],
                    [ math.sin(theta), math.cos(theta) , 0 ],
                    [ 0           , 0            , 1 ]])

def Ry(theta):
    return np.matrix([[ math.cos(theta), 0, math.sin(theta)],
                    [ 0           , 1, 0           ],
                    [-math.sin(theta), 0, math.cos(theta)]])

def Rx(theta):
    return np.matrix([[ 1, 0           , 0           ],
                    [ 0, math.cos(theta),-math.sin(theta)],
                    [ 0, math.sin(theta), math.cos(theta)]])

def modify_points(pt, type, axis, value, xmax, xmin, zmax, zmin, ymax, ymin):
    if type == "translate":
        if "x" in axis:
            axis2 = 0
        elif "z" in axis:
            axis2 = 2
        elif "y" in axis:
            axis2 = 1
        
        if "+" in axis:
            pt[axis2] += value
        else:
            pt[axis2] -= value
    elif type == "scale":
        if "x" in axis:
            if "+" in axis:
                pt[0] = (pt[0] - xmin) * value + xmin
            else:
                pt[0] = (pt[0] - xmax) * value + xmax
        elif "z" in axis:
            if "+" in axis:
                pt[2] = (pt[2] - zmin) * value + zmin
            else:
                pt[2] = (pt[2] - zmax) * value + zmax
        else:
            if "+" in axis:
                pt[1] = (pt[1] - ymin) * value + ymin
            else:
                pt[1] = (pt[1] - ymax) * value + ymax

    else:
        point = [(xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2]
        value = -value
        if "-x" in axis:
            rot_mat = Rx(-value)
        if "+x" in axis:
            rot_mat = Rx(value)
        if "+y" in axis:
            rot_mat = Ry(-value)
        if "-y" in axis:
            rot_mat = Ry(value)

        pt -= point
        pt = (rot_mat @ pt).tolist()[0]
        pt += np.array(point)

    return pt

parser = argparse.ArgumentParser(description='Point Cloud Registration')
parser.add_argument('--category', type=str, default='fridge', metavar='N',
                    help='Name of the category')
parser.add_argument('--split', type=str, default='train', metavar='N',
                    choices=['train', 'test'],
                    help='train or test')
args = parser.parse_args()

root_dir = os.path.join("../data/%s/shapes"%args.category, "%s_before"%args.split)

try:
    root_dir2 = root_dir.replace("_before", "")
    os.mkdir(root_dir2)
    
except:
    pass

for shape in os.listdir(root_dir):
    if "txt" in shape: continue
    seg_file = np.load(open(os.path.join(root_dir, shape, "instance_segmentation.npy"), "rb"))
    pts1 = np.load(open(os.path.join(root_dir, shape, "new", "0.npy"), "rb"))
    
    answer_file = os.path.join(root_dir.replace('shapes', "choices").replace("_before", ""), shape)

    data = json.load(open(answer_file))
    choices = data["choices"]
    answer = data["answer"]

    for (i,choice) in enumerate(choices):

        try:
            choice_dir = "%s/%s"%(root_dir2, shape)+"_%d"%i
            os.mkdir(choice_dir)
            
        except:
            pass
        
        pts_1 = []

        seg_file = np.load(open(os.path.join(root_dir, shape, "instance_segmentation.npy"), "rb"))
        pts1 = np.load(open(os.path.join(root_dir, shape, "new", "0.npy"), "rb"))

        root_num = np.load(open(os.path.join(root_dir, shape, "root_num.npy"), "rb"))
        
        with open(os.path.join(choice_dir, "pc.ply"), "w") as file:
            for cpoint in pts1:
                pts_1.append("%f %f %f\n"%(cpoint[0],cpoint[1], cpoint[2]))

            file.write('''ply
                format ascii 1.0
                element vertex %d
                property float x
                property float y
                property float z
                end_header
                %s
            '''%(len(pts_1),"".join(pts_1)))
            file.close()
        
        if choice == "functional":
            type = "translate"
            mask = 1
            axis = "+x"
            value = 0
        else:
            type, mask, axis, value = choice.strip().split()

            if type == "scale" and mask == "1" and "y" in axis:
                if "-" in axis: axis = axis.replace("-", "+")
                else: axis = axis.replace("+", "-")
            # type = "translate"
            # mask = len(root_num)-1
            # axis = "+z"
            # value = choice

        mask = int(mask) + 1
        
        root = root_num[mask]
        root_label = seg_file[root-16]
        value = abs(float(value))
        pts1_revised = np.zeros_like(pts1)
        obj_points = []

        for (j,pt) in enumerate(pts1):
            if seg_file[j-16] == root_label:
                obj_points.append(pt)

        obj_point = np.array(obj_points)

        xmax = np.max(obj_point[:,0])
        xmin = np.min(obj_point[:,0])
        zmax = np.max(obj_point[:,2])
        zmin = np.min(obj_point[:,2])
        ymax = np.max(obj_point[:,1])
        ymin = np.min(obj_point[:,1])

        for (j,pt) in enumerate(pts1):
            pts1_revised[j] = pts1[j].copy()
            if seg_file[j-16] == root_label:
                pt_revised = modify_points(pt, type, axis, value, xmax, xmin, zmax, zmin, ymax, ymin)
                pts1_revised[j] = pt_revised

        pts = []
        with open(os.path.join(choice_dir, "pc_revised.ply"), "w") as file:
            for cpoint in pts1_revised:
                pts.append("%f %f %f\n"%(cpoint[0],cpoint[1], cpoint[2]))

            file.write('''ply
                format ascii 1.0
                element vertex %d
                property float x
                property float y
                property float z
                end_header
                %s
            '''%(len(pts),"".join(pts)))
            file.close()           
        
        with open(os.path.join(choice_dir, "answer.json"), "w") as file:
            d = {"choice": choice, "correct": i == answer}
            json.dump(d, file)

        with open(os.path.join(choice_dir, "revised_0.npy"), "wb") as file:
            np.save(file, pts1_revised)      
        
        segmented_dict = [[], [], [], [], [j for j in range(16)]]

        instance_idx = []
        instance_idx.append(0)
        clusters = []
        root_num2 = []
        root_des_radius = []
        spacing = 0.05
        dt = 0.01

        for (j,label) in enumerate(seg_file):
            segmented_dict[label].append(j+16)

        n_objects = 0
        count = 0
        material = []
        for seg in segmented_dict:
            if len(seg):
                # fluid = False
                # for j in seg:
                #     if (pts1_revised[j][0] ** 2 + pts1_revised[j][2] ** 2 + pts1_revised[j][1] ** 2) < 0.03:
                #         find_fluid = True
                #         fluid = True
                # if fluid:
                #     material.append('fluid')
                # else:
                material.append('rigid')

                count += len(seg)
                n_objects += 1
                instance_idx.append(count)
                clusters.append([[np.array([0]* len(seg), dtype=np.int32)]])
                root_num2.append([1])
                root_des_radius.append([spacing])

        try:
            os.mkdir(os.path.join(choice_dir, "arranged_points"))
        except:
            pass
        
        instance_idx.pop()

        pts_new = []
        for seg in segmented_dict:
            for s in seg:
                pts_new.append(pts1_revised[s])
        pts_new = np.array(pts_new)

        with open(os.path.join(choice_dir,"arranged_points", "%d.npy"%0), "wb") as f:
            np.save(f, pts_new)

        phases_dict = dict()
        phases_dict["instance_idx"] = instance_idx
        phases_dict["root_des_radius"] = root_des_radius
        phases_dict["root_num"] = root_num2
        phases_dict["clusters"] = clusters
        phases_dict["time_step"] = 10
        phases_dict["n_objects"] = n_objects
        phases_dict["n_particles"] = 2048
        phases_dict["dt"] = dt
        phases_dict["material"] = material

        with open(os.path.join(choice_dir,"phases_dict.pkl"), "wb") as f:
            pickle.dump(phases_dict, f)

