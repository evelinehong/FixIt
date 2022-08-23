import xml.etree.ElementTree as ET
import random
import os
import numpy as np
import shutil
import time
import warnings
import math
import json
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


def modify_urdf(in_dir, out_dir, value, axis, origin, ori_origin, ori_origin2, rot_min_txt, rot_max_txt, rot_min, rot_max, new_rot):
    new_origin = origin.copy()

    # if axis == 0:
    new_origin[axis] += value
    # else:
    #     new_origin[axis] -= value

    new_rot_txt = str(rot_min + new_rot)
    new_rot2_txt = str(rot_max + new_rot)
    new_origin2 = [-new_origin[0], -new_origin[1], -new_origin[2]]
    
    new_origin = map(str, new_origin)
    new_origin = ' '.join(new_origin)

    new_origin2 = map(str, new_origin2)
    new_origin2 = ' '.join(new_origin2)

    original_urdf = open(os.path.join(in_dir, "mobility_vhacd.urdf"))
    mobility = original_urdf.read()
    mobility = mobility.replace(ori_origin, new_origin)
    mobility = mobility.replace(ori_origin2, new_origin2)
    mobility = mobility.replace("lower=\"%s\""%rot_min_txt, "lower=\"%s\""%new_rot_txt)
    mobility = mobility.replace("upper=\"%s\""%rot_max_txt, "upper=\"%s\""%new_rot2_txt)
    new_urdf = open(os.path.join(out_dir, "mobility_vhacd.urdf"), "w")
    new_urdf.write(mobility)

def load_obj2(v, f):

    vertices = []; faces = []; maxx = -10000; minx = 10000; maxz = -10000; minz = 10000; maxy = -10000; miny = 10000
    up_x = []
    low_x = []

    for vertice in v:
        # if abs(vertice[2]) < 0.1:
        if vertice[0] > maxx:
            maxx = vertice[0]
            up_x = vertice
        if vertice[0] < minx:
            minx = vertice[0]
            low_x = vertice

        # if abs(vertice[0]) < 0.1:
        if vertice[2] > maxy:
            maxy = vertice[2]
        if vertice[2] < miny:
            miny = vertice[2]

        if vertice[1] > maxz:
            maxz = vertice[1]
        if vertice[1] < minz:
            minz = vertice[1]

        vertices.append(vertice)

    return v, f, maxx, minx, maxz, minz, maxy, miny, up_x, low_x

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
    return rot_mat

shapes = os.listdir("original")
for shape_id in shapes:
    # try:
    # if not shape_id == '100435': continue
    if not shape_id in ['100435', '100462', '100465', '100468', '100472', '100484', '100486']: continue

    if shape_id in os.listdir("functional"): continue
    dir = os.path.join("original", shape_id)

    box = json.load(open(os.path.join(dir, "bounding_box.json")))
    ground = box["min"][1]

    tree = ET.parse(os.path.join(dir, 'mobility_vhacd.urdf'))
    root = tree.getroot()

    collision = []
    for link in root:
        if link.tag == "joint" and link.attrib["type"] == "revolute":
            origin = link[0].attrib["xyz"]
            origin = list(map(float, origin.split()))
            ori_origin = link[0].attrib["xyz"]
            axis = link[1].attrib["xyz"]
            axis = list(map(float, axis.split()))
            rot_min_txt = link[4].attrib["lower"]
            rot_max_txt = link[4].attrib["upper"]
            rot_min = float(link[4].attrib["lower"])
            rot_max = float(link[4].attrib["upper"])
            
            rot = (rot_min + rot_max) / 2

            child = link[2].attrib["link"]

            for link2 in root:
                if link2.attrib['name'] == child:
                    part_v_list = []

                    for link_child in link2:
                        if link_child.tag == 'visual':
                            obj_file = link_child[1][0].attrib["filename"]
                            v, _ = load_obj(os.path.join(dir, obj_file))
                            part_v_list.append(v)
                            ori_origin2 = link_child[0].attrib["xyz"]
                        if link_child.tag == 'collision':
                            collision.append(link_child[1][0].attrib["filename"])
                    
                    part_v = np.vstack(part_v_list)

                    zmin_index = np.argmax([abs(v[2] - 0) for v in part_v])
                    zmin = part_v[zmin_index]

    result = json.load(open(os.path.join(dir, "result.json")))
    children = result[0]["children"]
    for child in children:
        if child['name'] == 'handle':
            handle = [os.path.join("textured_objs", obj + ".obj") for obj in child["objs"]]
        else:
            body = [os.path.join("textured_objs", obj + ".obj") for obj in child["objs"]]

    root_v_list = []; root_f_list = []; tot_v_num = 0;
    for obj in body:
        v, f = load_obj(os.path.join(dir, obj))
        root_v_list.append(v);
        root_f_list.append(f+tot_v_num);
        tot_v_num += v.shape[0];

    root_v = np.vstack(root_v_list)
    root_f = np.vstack(root_f_list)

    v, f, maxx, minx, maxz, minz, maxy, miny, up_x, low_x = load_obj2(root_v, root_f)

    root_v_list = []; root_f_list = []; tot_v_num = 0;

    for obj in handle:
        v, f = load_obj(os.path.join(dir, obj))

        v -= origin
        v = np.transpose(v)
        rot_mat = [  [1.0000000,  0.0000000,  0.0000000],
        [0.0000000, -0.6753328,  0.7375131],
        [0.0000000, -0.7375131, -0.6753328] ]
        rotation_axis = np.array([-1, 0, 0])
        rotation_vector = rot * rotation_axis

        rot_mat = get_rotation_matrix(rotation_axis, rot)

        v = rot_mat @ v
        v = np.transpose(v)
        v += origin

        root_v_list.append(v);
        root_f_list.append(f+tot_v_num);
        tot_v_num += v.shape[0];

    root_v = np.vstack(root_v_list)
    root_f = np.vstack(root_f_list)

    up_v = root_v - [maxx, 0, 0]
    up_idx = np.argmin([abs(v[0]) for v in up_v])
    up_v = root_v[up_idx]

    minx_v = root_v - [minx, 0, 0]
    minx_idx = np.argmin([abs(v[0]) for v in minx_v])
    minx_v = root_v[minx_idx]

    maxz2 = np.max([v[1] for v in root_v])
    minz2 = np.min([v[1] for v in root_v])
    z2_range = maxz2 - minz2

    max_valid_z = maxz - up_v[1] 
    min_valid_z = maxz - minz2 - 0.02

    # max_move_z = (minz - minz2) / 2
    max_move_z = (minz - zmin[1]) 
    min_move_z = max_valid_z - 0.1

    xrange = maxx - minx

    find_fixed = False

    os.mkdir("functional/%s" %shape_id)
    i = 0
    while i < 13:
        new_dir = "functional/%s/%d"%(shape_id, i)
        shutil.copytree(dir, new_dir)

        new_rot = random.uniform(rot_min+0.1, (rot - math.pi/2))

        for obj in handle:
            rot_mat = modify_obj2(os.path.join(new_dir, obj), 0, new_rot, axis, origin, keep=False)
        for obj in collision:
            rot_mat = modify_obj2(os.path.join(new_dir, obj), 0, new_rot, axis, origin, keep=False)
        for obj in handle:
            modify_obj("translate", os.path.join(new_dir, obj), 0, 0, 1, origin)
        for obj in collision:
            modify_obj("translate", os.path.join(new_dir, obj), 0, 0, 1, origin)

        modify_urdf(dir, new_dir, 0, 1, origin, ori_origin, ori_origin2, rot_min_txt, rot_max_txt, rot_min, rot_max, new_rot)
        zmin2  = zmin - origin
        zmin2 = rot_mat@zmin2
        zmin2 += origin
        hand_position = np.array([0, -zmin2[2] + 0.85, zmin2[1] - 0.25])

        time1 = time.time()
        # print (joint)
        keep = simulate(new_dir, hand_position, ground, maxx, minx, maxy, miny, maxz, "functional")
        if not keep:
            print ("not keep")
            cmd = "rm -rf %s"%"functional/%s/%d"%(shape_id, i)
            call(cmd, shell=True)
            i -= 1
        i += 1
    #     time2 = time.time()

    os.mkdir("malfunctional/%s" %shape_id)

    i = 0

    while i < 52:
        new_dir = "malfunctional/%s/%d"%(shape_id, i)
        shutil.copytree(dir, new_dir)
        os.mkdir(os.path.join(new_dir, "output"))

        move = random.uniform(max_move_z, min_move_z)

        root_v_list = []; root_f_list = []; tot_v_num = 0

        # if random.random() < 1.1:
        new_rot = random.uniform(rot_min+0.1, (rot - math.pi/2))
        # else:
        #     new_rot = random.uniform(rot + math.pi/2, rot_max)    
        for obj in handle:
            rot_mat = modify_obj2(os.path.join(new_dir, obj), 0, new_rot, axis, origin, keep=False)
        for obj in collision:
            rot_mat = modify_obj2(os.path.join(new_dir, obj), 0, new_rot, axis, origin, keep=False)
        for obj in handle:
            modify_obj("translate", os.path.join(new_dir, obj), move, 0, 1, origin)
        for obj in collision:
            modify_obj("translate", os.path.join(new_dir, obj), move, 0, 1, origin)

        # change_obj = random.choice(change_objs)
        # axis, root_v, origin, part_v_objs, points, joint, ori_origin, ori_origin2, rot, rot_min = change_obj
        # xmax, xmin, ymax, ymin, zmax, zmin = points

        # new_rot = random.uniform(rot_min+0.1, (rot - math.pi/2 - rot_min))
        # rotation_axis = np.array(axis)
        # rot_mat = get_rotation_matrix(rotation_axis, new_rot)

        # # if axis == 1:
        # v, f, maxx, minx, maxz, minz, maxy, miny, up_x, low_x = load_obj2(body[0])

        # fout = open(os.path.join(new_dir, "output", "answer.txt"), "w")       
        
        # type = "translate"
        # fout.write("%s\n"%type)
        # point = zmax
        # value = random.uniform(0.3, 0.9)
        # value = -value
        # fout.write("+z\n")
        
        # fout.close()

        modify_urdf(dir, new_dir, move, 1, origin, ori_origin, ori_origin2, rot_min_txt, rot_max_txt, rot_min, rot_max, new_rot)  

        # for obj in part_v_objs:
        #     print (obj)
        #     modify_obj(type, os.path.join(new_dir, obj), value, point, rot_mat, origin)

        zmin2  = zmin - origin
        zmin2 = rot_mat@zmin2
        zmin2 += origin
        zmin2[1] += move
        hand_position = np.array([0, -zmin2[2] + 0.85, zmin2[1] - 0.25])

        # time1 = time.time()
        # print (joint)
        keep = simulate(new_dir, hand_position, ground, maxx, minx, maxy, miny, maxz, "malfunctional")
        
        if not keep:
            print ("not keep")
            cmd = "rm -rf %s"%"malfunctional/%s/%d"%(shape_id, i)
            call(cmd, shell=True)
            continue
            # i -= 1
        # time2 = time.time()

        new_up_v_1 = up_v[1] + move
        new_minz2 = minz2 + move
        
        valid_action_min = (max_valid_z - move) 
        valid_action_max = (min_valid_z - move) 

        output = [valid_action_min, valid_action_max]

        invalid_1_max = valid_action_min - 0.1
        invalid_1_min = 1.0

        output2 = [invalid_1_min, invalid_1_max]

        invalid_2_max = 6.0
        invalid_2_min = valid_action_max + 0.1
        
        output3 = [invalid_2_min, invalid_2_max]

        new_rot_cal = math.pi - rot + new_rot
        rot_sym_z = "-z"
        rot_sym_y = "-y"

        ori_y = up_v[1] - minz2
        ori_x = ori_y * abs(math.sin(new_rot_cal))
        new_y = maxz - new_minz2
        new_z = ori_y * abs(math.cos(new_rot_cal))
        new_x = math.sqrt(new_y ** 2 - new_z ** 2)
        scale = new_x / ori_x

        scale_max = scale + 3
        scale_min = 1.1

        output4 = [x for x in range(math.ceil(scale), math.floor(scale_max) + 1)]

        output5 = [x for x in range(math.ceil(scale_min), math.floor(scale) + 1)]

        ori_y = up_v[1] - minz2
        ori_x = ori_y * math.sin(math.pi - rot)
        new_y = maxz - new_minz2
        new_z = ori_y * math.cos(math.pi - rot)
        new_z = math.sqrt(new_y ** 2 - ori_x ** 2)

        scale = new_x / new_z

        scale_max = scale + 3
        scale_min = 1.1

        output6 = [x for x in range(math.ceil(scale), math.floor(scale_max) + 1)]

        output7 = [x for x in range(math.ceil(scale_min), math.floor(scale) + 1)]
        
        import pickle
        data = [move, new_rot, output, output2, output3, output4, output5, output6, output7, maxx, minx, maxy, miny, maxz, minz, hand_position]
        with open (os.path.join(new_dir, "answer_dict.json"), "wb") as fout:
            pickle.dump(data, fout)
        # with open (os.path.join(new_dir, "actions.txt"), "w") as fout:
        #     fout.write("initial translation - z:\n")
        #     fout.write(str(move)+"\n")
        #     fout.write("initial rot:\n")
        #     fout.write(str(new_rot)+"\n")
        #     fout.write("valid translation + z:\n")
        #     fout.write(str(output)+"\n")
        #     fout.write("invalid translation + z:\n")
        #     fout.write(str(output2)+"\n")

        #     fout.write("invalid translation + z:\n")
        #     fout.write(str(output3)+"\n")

        #     fout.write("valid scale %s:\n"%rot_sym_y)
        #     fout.write(str(output4)+"\n")
        #     fout.write("invalid scale %s:\n"%rot_sym_y)
        #     fout.write(str(output5)+"\n")

        #     fout.write("valid scale %s:\n"%rot_sym_z)
        #     fout.write(str(output6)+"\n")
        #     fout.write("invalid scale %s:\n"%rot_sym_z)
        #     fout.write(str(output7)+"\n")



        i += 1
    #             cmd = "blender --background --python render.py -- --shape %s > /dev/null"%new_dir
    #             print (cmd)
    #             call(cmd, shell=True)
    #             time3 = time.time()
    #             print ("rendering costs %.2fs using gpu" % (time3 - time2))
    #             # print ("simulation costs %.2fs" % (time2 - time1))
    #     except:
    #         pass
    # # except:
    #     pass

            
        




            





                        

                        