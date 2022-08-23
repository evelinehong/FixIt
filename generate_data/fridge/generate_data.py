import xml.etree.ElementTree as ET
import random
import os
import numpy as np
import shutil
import time
import warnings
import math
import json
from numpy.random import choice

warnings.filterwarnings("ignore")
from simulate import simulate
from subprocess import call

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

def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []; faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    f = np.vstack(faces)
    v = np.vstack(vertices)

    chosen_v = []

    for face in f:
        for idx in face:
            chosen_v.append(v[idx-1])

    v = np.array(chosen_v)

    return v, f

def modify_urdf(type, in_dir, out_dir, value, axis, origin, ori_origin, ori_origin2, point):
    new_origin = origin.copy()

    # if axis == 0:
    new_origin[axis] += value
    # else:
    #     new_origin[axis] -= value

    if type == 'rotate':
        new_origin = point

    new_origin2 = [-new_origin[0], -new_origin[1], -new_origin[2]]  

    new_origin = map(str, new_origin)
    new_origin = ' '.join(new_origin)

    new_origin2 = map(str, new_origin2)
    new_origin2 = ' '.join(new_origin2)

    original_urdf = open(os.path.join(in_dir, "mobility.urdf"))    
    if type != 'rotate':
        mobility = original_urdf.read()
        mobility = mobility.replace(ori_origin, new_origin)
        mobility = mobility.replace(ori_origin2, new_origin2)
        new_urdf = open(os.path.join(out_dir, "mobility.urdf"), "w")
        new_urdf.write(mobility)
    else:
        if axis == 2:
            axis = 0
        else:
            axis = 2
        lines = original_urdf.readlines()
        lines2 = lines
        for (i,line) in enumerate(lines):
            if ori_origin in line:
                lines2[i] = lines2[i].replace(ori_origin, new_origin)
                point = [0, math.cos(value), 0]
                point[axis] += -math.sin(value)
                if "-" in lines[i+1]:
                    point = [0, -math.cos(value), 0]
                    point[axis] += math.sin(value)
                point = map(str, point)
                point = ' '.join(point)
                lines2[i+1] = "		<axis xyz=\"%s\"/>"%point
            if ori_origin2 in line:
                lines2[i] = lines2[i].replace(ori_origin2, new_origin2)
        new_urdf = open(os.path.join(out_dir, "mobility.urdf"), "w")
        new_urdf.writelines(lines2)
    
def get_rotation_matrix(axis, theta):
    """
    Find the rotation matrix associated with counterclockwise rotation
    about the given axis by theta radians.
    Credit: http://stackoverflow.com/users/190597/unutbu

    Args:
        axis (list): rotation axis of the form [x, y, z]
        theta (float): rotational angle in radians

    Returns:
        array. Rotation matrix.
    """

    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]]) 

def modify_obj(type, fn, value, point, axis, origin):
    fin = open(fn.replace(".obj", "_copy.obj"), 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    fout = open(fn, 'w')

    if type == "rotate":
        if axis == 0:
            rot_mat = Rx(value)
        else:
            rot_mat = Rz(value)

    for line in lines:
        if line.startswith('v '):
            vertice = np.float32(line.split()[1:4])

            if type == "scale":
                vertice[axis] = (vertice[axis] - point) * value + point
            if type == "translate":
                vertice[axis] = vertice[axis] + value
            if type == "rotate":
                vertice -= point
                vertice = (rot_mat @ vertice).tolist()[0]
                vertice += np.array(point)

            fout.write('v %f %f %f\n' % (vertice[0], vertice[1], vertice[2]))
        else:
            fout.write(line + "\n")

    fout.close()

def modify_obj2(fn, value, rot_max, axis,  origin, keep=False):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    if keep:
        fout = open(fn, 'w')
    else:
        fout = open(fn.replace(".obj", "_copy.obj"), "w")
        
    rot_mat = get_rotation_matrix(axis, value+rot_max)

    for line in lines:
        if line.startswith('v '):
            vertice = np.float32(line.split()[1:4])

            vertice -= origin
            vertice = (rot_mat @ vertice).tolist()
            vertice += np.array(origin)

            fout.write('v %f %f %f\n' % (vertice[0], vertice[1], vertice[2]))
        else:
            fout.write(line + "\n")

    fout.close()

shapes = os.listdir("original")
for shape_id in shapes:
    # try:
    # if shape_id != '10068': continue
    # print (shape_id)
    if shape_id in os.listdir("functional"): continue
    dir = os.path.join("original", shape_id)

    tree = ET.parse(os.path.join(dir, 'mobility.urdf'))
    root = tree.getroot()

    change_objs = []
    joints = []
    j = 0

    find_fixed = False

    for link in root:
        if link.tag == "joint":
            if link.attrib["type"] in ["fixed"]:
                j = 0
                find_fixed = True

            if link.attrib["type"] in ["revolute"]:
                joints.append(j)
                child = link[2].attrib["link"]
                ori_origin = link[0].attrib["xyz"]
                origin = list(map(float, ori_origin.split()))
                axis = link[1].attrib["xyz"]
                axis = list(map(float, axis.split()))
                axis2 = axis.copy()
                axis = np.nonzero(axis)[0][0]

                rot_min = float(link[4].attrib["lower"])
                rot_max = float(link[4].attrib["upper"])

                for link2 in root:
                    if link2.attrib['name'] == child:
                        part_v_list = []
                        part_v_objs = []

                        for link_child in link2:
                            if link_child.tag == 'visual':
                                obj_file = link_child[1][0].attrib["filename"]
                                part_v_objs.append(obj_file)
                                v, _ = load_obj(os.path.join(dir, obj_file))
                                part_v_list.append(v)
                                ori_origin2 = link_child[0].attrib["xyz"]

                        part_v = np.vstack(part_v_list)
                        xmax = np.max([x[0] for x in part_v])
                        xmin = np.min([x[0] for x in part_v])
                        ymax = np.max([x[2] for x in part_v])
                        ymin = np.min([x[2] for x in part_v])
                        zmax = np.max([x[1] for x in part_v])
                        zmin = np.min([x[1] for x in part_v])
                        points = [xmax, xmin, ymax, ymin, zmax, zmin]

                        if xmax - xmin > 0.25:
                            if not find_fixed:
                                change_objs.append([axis, origin, part_v_objs, points, j + 1, ori_origin, ori_origin2, rot_min, axis2])
                            else:
                                change_objs.append([axis, origin, part_v_objs, points, j, ori_origin, ori_origin2, rot_min, axis2])
            j += 1

    if not len(change_objs):
        continue


    joints = [obj[4] for obj in change_objs]

    os.mkdir("functional/%s" %shape_id)
    for i in range(6):
        new_dir = "functional/%s/%d"%(shape_id, i)
        shutil.copytree(dir, new_dir)
        os.mkdir(os.path.join(new_dir, "output"))

        change_idx = random.choice(range(len(change_objs)))
        change_obj = change_objs[change_idx]
        axis, origin, part_v_objs, points, joint, ori_origin, ori_origin2, rot_min, axis2 = change_obj
        xmax, xmin, ymax, ymin, zmax, zmin = points

        joint_limit = 0.0

        rot_max = random.uniform(0.75, math.pi)

        other = False

        for obj in part_v_objs:
            modify_obj2(os.path.join(new_dir, obj), rot_min, rot_max, axis2, origin, keep=True)

        root_v_list = []
        for obj in part_v_objs:
            v, f = load_obj(os.path.join(new_dir, obj))
            root_v_list.append(v);

        part_v = np.vstack(root_v_list)

        xmax = np.max([x[0] for x in part_v])
        xmin = np.min([x[0] for x in part_v])
        ymax = np.max([x[2] for x in part_v])
        ymin = part_v[np.argmax([x[2] for x in part_v])]
        ymin2 = np.min([x[2] for x in part_v])
        zmax = np.max([x[1] for x in part_v])
        zmin = np.min([x[1] for x in part_v])

        if axis == 1:
            pt = [ymin[0], -ymin[2], (zmax+zmin)/2]
        else:
            pt = [(xmax+xmin)/2, -ymin[2], ymin[1]]

        time1 = time.time()
        # print (joint)
        simulate(new_dir, joint, joint_limit, rot_max, rot_min, pt)
        time2 = time.time()

        # cmd = "blender --background --python render.py -- --shape %s > /dev/null"%new_dir
        # print (cmd)
        # call(cmd, shell=True)
        # time3 = time.time()
        # print ("rendering costs %.2fs using gpu" % (time3 - time2))
        print ("simulation costs %.2fs" % (time2 - time1))

    os.mkdir("malfunctional/%s" %shape_id)
    
    for i in range(24):
        new_dir = "malfunctional/%s/%d"%(shape_id, i)
        shutil.copytree(dir, new_dir)
        os.mkdir(os.path.join(new_dir, "output"))

        change_idx = random.choice(range(len(change_objs)))
        change_obj = change_objs[change_idx]
        axis, origin, part_v_objs, points, joint, ori_origin, ori_origin2, rot_min, axis2 = change_obj
        part_v_objs2 = part_v_objs.copy()
        xmax, xmin, ymax, ymin, zmax, zmin = points
        original_axis = axis.copy()
        # if axis == 1:
        fout = open(os.path.join(new_dir, "output", "data.json"), "w")
        
        # if random.random() > 0.5:
        #     rot_max = random.uniform(2.5, math.pi)
        # else:
        #     rot_max = random.uniform(0.75, 1.0)
        type = choice(["scale", "translate", "rotate"], 1,  p=[0.35, 0.35, 0.3])[0]
        # type = "rotate"
        rot_max = random.uniform(0.75, math.pi)

        if type == "rotate":
            if random.random() < 0.5:
                rot_max = math.pi/2
            else:
                rot_max = math.pi

        # for obj in part_v_objs:
        #     modify_obj2(os.path.join(new_dir, obj), rot_min, rot_max, axis2, origin)
        
        other = False

        if len(change_objs) == 2 and random.random() < 0.5:
            other = True
            type = random.choice(["scale", "translate"])
            rot_max = random.uniform(0.75, math.pi)

            for obj in part_v_objs:
                modify_obj2(os.path.join(new_dir, obj), rot_min, rot_max, axis2, origin, keep=True)

            for c in range(len(change_objs)):
                if c != change_idx:
                    axis, origin, part_v_objs, points, _, ori_origin, ori_origin2, rot_min, axis2 = change_objs[c]
            for obj in part_v_objs:
                modify_obj2(os.path.join(new_dir, obj), rot_min, 0, axis2, origin)
            rot_max2 = rot_max
        else:
            for obj in part_v_objs:
                modify_obj2(os.path.join(new_dir, obj), rot_min, rot_max, axis2, origin)
                rot_max2 = rot_max

        root_v_list = []
        for obj in part_v_objs:
            v, f = load_obj(os.path.join(new_dir, obj.replace(".obj", "_copy.obj")))
            root_v_list.append(v);

        part_v = np.vstack(root_v_list)
            # rot_max = math.pi

        xmax = np.max([x[0] for x in part_v])
        xmin = np.min([x[0] for x in part_v])
        ymax = np.max([x[2] for x in part_v])
        ymin = np.min([x[2] for x in part_v])
        zmax = np.max([x[1] for x in part_v])
        zmin = np.min([x[1] for x in part_v])
        
        # fout.write("%s\n"%type)

        if type == "scale":
            if random.random() < 0.75 or other:
                if axis == 1 and len(change_objs) >= 2 and abs(change_objs[1][1][0] - change_objs[0][1][0]) < 0.25:
                    axis = 1
                    point = zmax
                    axis_w = "-z"
                    # fout.write("-z\n")

                elif axis == 1:
                    if abs((rot_max) - math.pi) < 0.5 or (abs((rot_max) - math.pi/2) > 0.5 and random.random() > 0.5) or rot_max < 0.5:
                        axis = 0
                        if abs(xmax - origin[0]) > abs(xmin - origin[0]):
                            point = xmin
                            axis_w = "+x"
                            # fout.write("+x\n")
                        else:
                            point = xmax
                            axis_w = "-x"
                    else:
                        axis = 2
                        # if abs(xmax - origin[0]) > abs(xmin - origin[0]):
                        #     point = ymin
                        #     axis_w = "+y"
                        #     # fout.write("+x\n")
                        # else:
                        point = ymin
                        axis_w = "-y"                    
                        # fout.write("-x\n")
                elif axis == 0:
                    if abs((rot_max) - math.pi) < 0.5 or (abs((rot_max) - math.pi/2) > 0.5 and random.random() > 0.5) or rot_max < 0.5:
                        axis = 1
                        if abs(zmax + origin[1]) > abs(zmin + origin[1]):
                            point = zmin
                            axis_w = "+z"
                            # fout.write("+z\n")
                        else:
                            point = zmax
                            axis_w = "-z"
                    else:
                        axis = 2
                        # if abs(xmax - origin[0]) > abs(xmin - origin[0]):
                        #     point = ymin
                        #     axis_w = "+y"
                        #     # fout.write("+x\n")
                        # else:
                        point = ymax
                        axis_w = "-y"  
                        # fout.write("-z\n")
                
                if random.random() > 0.5 and len(change_objs) >= 2:
                    value = random.uniform(1.1, 1.25)
                    joint_limit = 0.1
                    # fout.write("smaller\n")
                else:
                    print ("5") 
                    value = random.uniform(0.5, 0.8)
                    joint_limit = 0.0
                    # fout.write("larger\n")

            else:
                if axis == 1:
                    axis = 1
                    print ('6')
                    if random.random() < 0.5:
                        point = zmin
                        axis_w = "+z"
                        # fout.write("+z\n")
                    else:
                        point = zmax
                        axis_w = "-z"
                        # fout.write("-z\n")

                elif axis == 0:
                    print ('7')
                    axis = 0
                    if random.random() < 0.5:
                        point = xmin
                        axis_w = "+x"
                        # fout.write("+x\n")
                    else:
                        point = xmax
                        axis_w = "-x"
                        # fout.write("-x\n")
                
                value = random.uniform(0.5, 0.8)
                joint_limit = 0.0
                # fout.write("larger\n")
            # fout.write(str(1/value))
        elif type == "translate":
    #     elif rand < 0.67:
    #         type = "translate"
    #         fout.write("%s\n"%type)
            
            if random.random() < 0.75 or other:
                if axis == 1:
                    axis = 0
                    value = random.uniform(max((xmax - xmin)/4, 0.1), min(0.3, max(0.1,(xmax - xmin)/2)))
                    if (abs(xmax - origin[0]) - abs(xmin - origin[0])) > 0:
                        point = xmin
                        axis_w = "-x"
                        if rot_max - math.pi/2 > 0:
                            value = -value
                            axis_w = "+x"
                        # fout.write("-x\n")
                    else:
                        value = -value
                        point = xmax
                        axis_w = "+x"
                        if rot_max - math.pi/2 > 0:
                            value = -value
                            axis_w = "-x"
                        # fout.write("+x\n")
                
                elif axis == 0:
                    axis = 1
                    value = random.uniform(max((zmax - zmin)/8, 0.1), min(0.3, max(0.1,(zmax - zmin)/4)))
                    if (abs(zmax - origin[1]) - abs(zmin - origin[1])) * (rot_max - math.pi/2) < 0:
                        point = zmin
                        axis_w = "-z"
                        # fout.write("-z\n")
                    else:
                        value = -value
                        point = zmax
                        axis_w = "+z"
                        # fout.write("+z\n") 

                joint_limit = 0.1
            
            else:
                if axis == 1:
                    axis = 1
                    if random.random() < 0.5:
                        value = random.uniform(max((zmax - zmin)/8, 0.08), min(0.3, max(0.1,(zmax - zmin)/4)))
                        point = zmin
                        axis_w = "-z"
                        # fout.write("-z\n")
                    else:
                        value = random.uniform(max((zmax - zmin)/8, 0.08), min(0.3, max(0.1,(zmax - zmin)/4)))
                        value = -value
                        point = zmax
                        axis_w = "+z"
                        # fout.write("+z\n")

                else:
                    axis = 0
                    value = random.uniform(max((xmax - xmin)/4, 0.08), min(0.3, max(0.1,(xmax - xmin)/2)))
                    if random.random() < 0.5:
                        point = xmin
                        axis_w = "-x"
                        # fout.write("-x\n")
                    else:
                        value = -value
                        point = xmax
                        axis_w = "+x"
                        # fout.write("+x\n")

                joint_limit = 0.0            

            if not other:
                modify_urdf(type, dir, new_dir, value, axis, origin, ori_origin, ori_origin2, point)            
            # fout.write(str(-value))

        else:
            # type = "rotate"
            if random.random() > 0.5:
                value = random.uniform(0.3, 0.5)
            else:
                value = random.uniform(-0.5, -0.3)
            if abs(xmax - origin[0]) > abs(xmin - origin[0]):
                point = [(xmax+xmin)/2, (zmax+zmin)/2, (ymax+ymin)/2]
            else:
                point = [(xmax+xmin)/2, (zmax+zmin)/2, (ymax+ymin)/2]

            if axis == 1:
                if rot_max == math.pi/2:
                    axis = 2
                    axis_w = "y"
                    joint_limit = 0.0
                else:
                    axis = 0
                    axis_w = "x"
                    joint_limit = 0.0
                # if random.random() < 0.5:
                #     axis = 2
                # else:
                #     axis = 0
                # if (axis == 2 and rot_max == math.pi) or (axis == 0 and rot_max == math.pi/2):
                #     axis_w = "y"
                #     joint_limit = 0.1
                # else:
                #     axis_w = "x"
                #     joint_limit = 0.0
            else:
                if rot_max == math.pi/2:
                    axis = 2
                    axis_w = "y"
                    joint_limit = 0.0
                else:
                    axis = 1
                    axis_w = "z"
                    joint_limit = 0.0
            
            value = -value
            if value > 0:
                if axis_w == "y":
                    axis_w = "+" + axis_w
                else:
                    axis_w = "-" + axis_w
            else:
                if axis_w == "y":
                    axis_w = "-" + axis_w
                else:
                    axis_w = "+" + axis_w
            
            # modify_urdf(type, dir, new_dir, value, axis, origin, ori_origin, ori_origin2, point)

        # def calculate_value(value, rot_max):
        #     print (math.sin(rot_max))
        #     print (value)
        #     alpha = math.asin(abs(math.sin(rot_max)) / value)
        #     x_value = math.sin(rot_max)/math.tan(alpha)
        #     ori_x_value = math.cos(rot_max)
        #     value_2 = x_value / ori_x_value

        #     return value_2
         
        # if axis != original_axis and "scale" in type:
        #     value_2 = calculate_value(value, rot_max)
        # else:
        # if "scale" in type:
        #     value_2 = 1/value_2
        # else:
        #     value_2 = -value_2

        #     value_2 = value

        # if (rot_max - math.pi/2) > 0:
        #     if "-" in axis_w:
        #         axis_w = axis_w.replace("-", "+")
        #     else:
        #         axis_w = axis_w.replace("+", "-")
        
    #     # fout.write("\n")
    #     # fout.write(str(part_v_objs) + "\n")
    #     # fout.write(str(rot_max))
    #     # fout.close()
        assert value != 0.0

        for obj in part_v_objs:
            modify_obj(type, os.path.join(new_dir, obj), value, point, axis, origin)

        root_v_list = []
        for obj in part_v_objs2:
            v, f = load_obj(os.path.join(new_dir, obj))
            root_v_list.append(v);

        part_v = np.vstack(root_v_list)

        xmax = np.max([x[0] for x in part_v])
        xmin = np.min([x[0] for x in part_v])
        ymax = np.max([x[2] for x in part_v])
        ymin = part_v[np.argmax([x[2] for x in part_v])]
        ymin2 = np.min([x[2] for x in part_v])
        zmax = np.max([x[1] for x in part_v])
        zmin = np.min([x[1] for x in part_v])
        
        if type != 'rotate':
            if original_axis == 1:
                pt = [ymin[0], -ymin[2], (zmax+zmin)/2]
            else:
                pt = [(xmax+xmin)/2, -ymin[2], ymin[1]]
        else:
            if rot_max == math.pi/2:
                pt = [(xmax+xmin)/2, -ymin[2], (zmax+zmin)/2]
            else:
                if abs(xmax - origin[0]) > abs(xmin - origin[0]):
                    pt = [xmax, (-ymax-ymin2)/2, (zmax+zmin)/2]
                else:
                    pt = [xmin, (-ymax-ymin2)/2, (zmax+zmin)/2]

        if type == 'scale' and 'x' in axis_w and not other:
            edge_1 = abs(math.sin(rot_max))
            edge_2 = abs(math.cos(rot_max))
            edge_2_new = edge_2 * value

            new_alpha = math.atan(edge_1/edge_2_new)
            if rot_max > math.pi/2:
                rot_max2 = math.pi - new_alpha
            else:
                rot_max2 = new_alpha
            rot_max = rot_max2

        if type == 'scale' and 'y' in axis_w and not other:
            edge_1 = abs(math.cos(rot_max))
            edge_2 = abs(math.sin(rot_max))
            edge_2_new = edge_2 * value

            new_alpha = math.atan(edge_2_new/edge_1)
            if rot_max > math.pi/2:
                rot_max2 = math.pi - new_alpha
            else:
                rot_max2 = new_alpha
            rot_max = rot_max2

        if type == "scale":
            value = 1 / value
        if type in ["translate", "rotate"]:
            value = abs(value)

        data = {"type": type, "axis": axis_w, "value": value, "objs": part_v_objs, "rot_max": rot_max, "rot_max2": rot_max2}
        json.dump(data, fout)

        time1 = time.time()
        # print (joint)
        simulate(new_dir, joint, joint_limit, rot_max2, rot_min, pt)
        time2 = time.time()

    #     # # cmd = "blender --background --python render.py -- --shape %s > /dev/null"%new_dir
    #     # # print (cmd)
    #     # # call(cmd, shell=True)
    #     # # time3 = time.time()
    #     # # print ("rendering costs %.2fs using gpu" % (time3 - time2))
    #     # print ("simulation costs %.2fs" % (time2 - time1))

    # # except:
    # #     pass

        
    




        





                        

                        