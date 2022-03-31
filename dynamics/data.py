import os
import torch
import time
import random
import numpy as np
import gzip
import pickle
import h5py
import copy


import multiprocessing as mp
import scipy.spatial as spatial
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.transform import Rotation as R

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from utils import rand_float, rand_int

import ipdb
_st=ipdb.set_trace

def collate_fn(data):
    return data[0]

def load_data(data_names, path, load_data_names=["obj_positions", "obj_rotations"]):

    if not isinstance(path, list):
        paths = [path]
        one_item = True
    else:
        paths = path
        one_item = False

    multiple_data = []
    for path in paths:
        data = []
        positions = np.load(open(path, "rb"))

        for data_name in data_names:
            if data_name == "positions":
                data.append(positions)

            elif data_name == "velocities":
                # should compute later on
                data.append(None)

            else:
                raise (ValueError, f"{data_name} not supported")
        multiple_data.append(data)

    if one_item:
        return multiple_data[0]

    return data

def recalculate_velocities(list_of_data, dt, data_names):
    positions_over_T = []
    velocities_over_T = []

    for data in list_of_data:
        positions = data[data_names.index("positions")]
        velocities = data[data_names.index("velocities")]
        positions_over_T.append(positions)
        velocities_over_T.append(velocities)

    output_list_of_data = []
    for t in range(len(list_of_data)):

        current_data = []
        for item in data_names:
            if item == "positions":
                current_data.append(positions_over_T[t])
            elif item == "velocities":
                if t == 0:
                    current_data.append(velocities_over_T[t])
                else:
                    current_data.append((positions_over_T[t] - positions_over_T[t - 1]) / dt)
            else:
                raise (ValueError, f"not supporting augmentation for {item}")
        output_list_of_data.append(current_data)

    return output_list_of_data

def init_stat(dim):
    # mean, std, count
    return np.zeros((dim, 3))

def normalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            stat[i][stat[i][:, 1] == 0, 1] = 1.
            s = Variable(torch.FloatTensor(stat[i]).cuda())

            stat_dim = stat[i].shape[0]
            n_rep = int(data[i].size(1) / stat_dim)
            data[i] = data[i].view(-1, n_rep, stat_dim)

            data[i] = (data[i] - s[:, 0]) / s[:, 1]

            data[i] = data[i].view(-1, n_rep * stat_dim)

    else:
        for i in range(len(stat)):
            stat[i][stat[i][:, 1] == 0, 1] = 1.

            stat_dim = stat[i].shape[0]

            n_rep = int(data[i].shape[1] / stat_dim)
            data[i] = data[i].reshape((-1, n_rep, stat_dim))

            data[i] = (data[i] - stat[i][:, 0]) / stat[i][:, 1]

            data[i] = data[i].reshape((-1, n_rep * stat_dim))

    return data

def denormalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            s = Variable(torch.FloatTensor(stat[i]).cuda())
            data[i] = data[i] * s[:, 1] + s[:, 0]
    else:
        for i in range(len(stat)):
            data[i] = data[i] * stat[i][:, 1] + stat[i][:, 0]

    return data

def visualize_neighbors(anchors, queries, idx, neighbors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(queries[idx, 0], queries[idx, 1], queries[idx, 2], c='g', s=80)
    ax.scatter(anchors[neighbors, 0], anchors[neighbors, 1], anchors[neighbors, 2], c='r', s=80)
    ax.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2], alpha=0.2)
    ax.set_aspect('equal')

    plt.show()


def find_relations_neighbor(positions, query_idx, anchor_idx, radius, order, var=False):
    if np.sum(anchor_idx) == 0:
        return []

    pos = positions.data.cpu().numpy() if var else positions


    point_tree = spatial.cKDTree(pos[anchor_idx])
    neighbors = point_tree.query_ball_point(pos[query_idx], radius, p=order)

    '''
    for i in range(len(neighbors)):
        visualize_neighbors(pos[anchor_idx], pos[query_idx], i, neighbors[i])
    '''

    relations = []
    for i in range(len(neighbors)):
        count_neighbors = len(neighbors[i])
        if count_neighbors == 0:
            continue

        receiver = np.ones(count_neighbors, dtype=np.int) * query_idx[i]
        sender = np.array(anchor_idx[neighbors[i]])

        # receiver, sender, relation_type
        relations.append(np.stack([receiver, sender, np.ones(count_neighbors)], axis=1))

    return relations


def make_hierarchy(attr, positions, velocities, idx, st, ed, phases_dict, count_nodes, clusters, verbose=0, var=False):
    order = 2
    n_root_level = len(phases_dict["root_num"][idx])
    attr, relations, relations_types, node_r_idx, node_s_idx, pstep = [attr], [], [], [], [], []

    relations_rev, relations_rev_types, node_r_idx_rev, node_s_idx_rev, pstep_rev = [], [], [], [], []

    pos = positions.data.cpu().numpy() if var else positions
    vel = velocities.data.cpu().numpy() if var else velocities

    for i in range(n_root_level):
        root_num = phases_dict["root_num"][idx][i]
        #root_sib_radius = phases_dict["root_sib_radius"][idx][i]
        root_des_radius = phases_dict["root_des_radius"][idx][i]
        root_pstep = phases_dict["root_pstep"][idx][i]

        ### relations between roots and desendants
        rels, rels_rev = [], []
        node_r_idx.append(np.arange(count_nodes, count_nodes + root_num))
        node_s_idx.append(np.arange(st, ed))
        node_r_idx_rev.append(node_s_idx[-1])
        node_s_idx_rev.append(node_r_idx[-1])
        pstep.append(1); pstep_rev.append(1)

        if verbose:
            centers = np.zeros((root_num, 3))
            # compute the mean of each sub-parts
            for j in range(root_num):
                des = np.nonzero(clusters[i][0]==j)[0] #indices inside the group
                center = np.mean(pos[st:ed][des, -3:], 0, keepdims=True)
                centers[j] = center[0]
                visualize_neighbors(pos[st:ed], center, 0, des)

        for j in range(root_num):
            desendants = np.nonzero(clusters[i][0]==j)[0]
            roots = np.ones(desendants.shape[0]) * j
            if verbose:
                print(roots, desendants)
            rels += [np.stack([roots, desendants, np.zeros(desendants.shape[0])], axis=1)]
            rels_rev += [np.stack([desendants, roots, np.zeros(desendants.shape[0])], axis=1)]
            if verbose:
                print(np.max(np.sqrt(np.sum(np.square(pos[st + desendants, :3] - centers[j]), 1))))

        relations.append(np.concatenate(rels, 0))
        relations_rev.append(np.concatenate(rels_rev, 0))
        relations_types.append("leaf-root")
        relations_rev_types.append("root-leaf")


        ### relations between roots and roots
        # point_tree = spatial.cKDTree(centers)
        # neighbors = point_tree.query_ball_point(centers, root_sib_radius, p=order)

        '''
        for j in range(len(neighbors)):
            visualize_neighbors(centers, centers, j, neighbors[j])
        '''

        rels = []
        node_r_idx.append(np.arange(count_nodes, count_nodes + root_num))
        node_s_idx.append(np.arange(count_nodes, count_nodes + root_num))
        pstep.append(root_pstep)

        # adding all possible pairs of root nodes
        roots = np.repeat(np.arange(root_num), root_num)
        siblings = np.tile(np.arange(root_num), root_num)
        if verbose:
            print(roots, siblings)
        rels += [np.stack([roots, siblings, np.zeros(root_num * root_num)], axis=1)]
        if verbose:
            print(np.max(np.sqrt(np.sum(np.square(centers[siblings, :3] - centers[j]), 1))))


        relations.append(np.concatenate(rels, 0))
        relations_types.append("root-root")

        ### add to attributes/positions/velocities
        positions = [positions]
        velocities = [velocities]
        attributes = []
        for j in range(root_num):
            ids = np.nonzero(clusters[i][0]==j)[0]
            if var:
                positions += [torch.mean(positions[0][st:ed, :][ids], 0, keepdim=True)]
                velocities += [torch.mean(velocities[0][st:ed, :][ids], 0, keepdim=True)]
            else:
                positions += [np.mean(positions[0][st:ed, :][ids], 0, keepdims=True)]
                velocities += [np.mean(velocities[0][st:ed, :][ids], 0, keepdims=True)]

            attributes += [np.mean(attr[0][st:ed, :][ids], 0, keepdims=True)]

        attributes = np.concatenate(attributes, 0)

        if not attributes[0, -1] == 0:
            import ipdb; ipdb.set_trace()
        assert(attributes[0, -1] == 0), "last dimension should save for parent node"
        attributes[0, -1] = 1


        if verbose:
            print('Attr sum', np.sum(attributes, 0))

        attr += [attributes]
        if var:
            positions = torch.cat(positions, 0)
            velocities = torch.cat(velocities, 0)
        else:
            positions = np.concatenate(positions, 0)
            velocities = np.concatenate(velocities, 0)

        # add #[root_num] of root nodes
        st = count_nodes
        ed = count_nodes + root_num
        count_nodes += root_num

        if verbose:
            print(st, ed, count_nodes, positions.shape, velocities.shape)

    attr = np.concatenate(attr, 0)
    if verbose:
        print("attr", attr.shape)

    relations += relations_rev[::-1]
    relations_types += relations_rev_types[::-1]

    node_r_idx += node_r_idx_rev[::-1]
    node_s_idx += node_s_idx_rev[::-1]
    pstep += pstep_rev[::-1]

    return attr, positions, velocities, count_nodes, relations, relations_types, node_r_idx, node_s_idx, pstep


def prepare_input(data, stat, args, phases_dict, verbose=0, var=False):

    # Arrangement:
    # particles, shapes, roots

    positions, velocities = data
    clusters = phases_dict["clusters"]

    n_shapes = 16

    count_nodes = positions.size(0) if var else positions.shape[0]
    n_particles = count_nodes - n_shapes

    if verbose:
        print("positions", positions.shape)
        print("velocities", velocities.shape)

        print("n_particles", n_particles)
        print("n_shapes", n_shapes)

    ### instance idx
    #   instance_idx (n_instance + 1): start idx of instance
    instance_idx = phases_dict["instance_idx"]
    if verbose:
        print("Instance_idx:", instance_idx)

    ### object attributes
    #   dim 10: [rigid, fluid, root_0, root_1, gripper_0, gripper_1, mass_inv,
    #            clusterStiffness, clusterPlasticThreshold, cluasterPlasticCreep]
    attr = np.zeros((count_nodes, args.attr_dim))

    ### construct relations
    Rr_idxs = []        # relation receiver idx list
    Rs_idxs = []        # relation sender idx list
    Ras = []            # relation attributes list
    values = []         # relation value list (should be 1)
    node_r_idxs = []    # list of corresponding receiver node idx
    node_s_idxs = []    # list of corresponding sender node idx
    psteps = []         # propagation steps

    ##### add env specific graph components
    rels = []
    rels_types = []

    pos = positions.data.cpu().numpy() if var else positions
    dis = pos[:n_particles, 1] - 0

    nodes = np.nonzero(dis < 0.1)[0]
    attr[-1, 2] = 1 # [0, 0, 1] is floor
    floor = np.ones(nodes.shape[0], dtype=np.int) * (n_particles + 0) #0 for idx starting from zero
    rels += [np.stack([nodes, floor, np.ones(nodes.shape[0])], axis=1)]

    if verbose and len(rels) > 0:
        print(np.concatenate(rels, 0).shape)

    nobjs = len(instance_idx) - 1
    phases_dict["root_pstep"] = [[args.pstep]]*nobjs

    ##### add relations between leaf particles

    # for model without hierarchy, just search particles that are close

    for i in range(n_shapes):
        attr[n_particles + i, 1] = 1

        pos = positions.data.cpu().numpy() if var else positions

        dis = np.sqrt(np.sum((pos[:n_particles] - pos[n_particles + i])**2, 1))

        nodes = np.nonzero(dis < 0.1)[0]

        if verbose:
            visualize_neighbors(positions, positions, 0, nodes)
            print(np.sort(dis)[:10])

        wall = np.ones(nodes.shape[0], dtype=np.int) * (n_particles + i)
        rels += [np.stack([nodes, wall, np.ones(nodes.shape[0])], axis=1)]

    queries = np.arange(n_particles)
    anchors = np.arange(n_particles)
    pos = positions
    pos = pos[:, -3:]
    
    for i in range(len(instance_idx) - 1):
        st, ed = instance_idx[i], instance_idx[i + 1]
        # attr[st:ed, 0] = 1

    for i in range(n_particles):
        attr[i, 0] = 1

    rels += find_relations_neighbor(pos, queries, anchors, args.neighbor_radius, 2, var)

    if verbose:
        print("Attr shape (after add env specific graph components):", attr.shape)
        print("Object attr:", np.sum(attr, axis=0))

    if len(rels) > 0:
        rels = np.concatenate(rels, 0)
        if rels.shape[0] > 0:
            if verbose:
                print("Relations neighbor", rels.shape)
            Rr_idxs.append(torch.LongTensor([rels[:, 0], np.arange(rels.shape[0])]))
            Rs_idxs.append(torch.LongTensor([rels[:, 1], np.arange(rels.shape[0])]))
            Ra = np.zeros((rels.shape[0], args.relation_dim))
            Ras.append(torch.FloatTensor(Ra))
            values.append(torch.FloatTensor([1] * rels.shape[0]))
            node_r_idxs.append(np.arange(n_particles))
            node_s_idxs.append(np.arange(n_particles + n_shapes))
            psteps.append(args.pstep)
            rels_types.append("leaf-leaf")

    if verbose:
        print('clusters', clusters)

    # add heirarchical relations per instance
    cnt_clusters = 0
    # clusters: [[[ array(#num_nodes_in_instance) ]]*n_root_level   ]*num_clusters

    if args.model_name not in ["GNS", "GNSRigid"]: #GNS has no hierarchy
        for i in range(len(instance_idx) - 1):
            st, ed = instance_idx[i], instance_idx[i + 1]

            n_root_level = len(phases_dict["root_num"][i])

            if n_root_level > 0:
                attr, positions, velocities, count_nodes, \
                rels, rels_type, node_r_idx, node_s_idx, pstep = \
                        make_hierarchy(attr, positions, velocities, i, st, ed,
                                       phases_dict, count_nodes, clusters[cnt_clusters], verbose, var)

                for j in range(len(rels)):
                    if verbose:
                        print("Relation instance", j, rels[j].shape)
                    Rr_idxs.append(torch.LongTensor([rels[j][:, 0], np.arange(rels[j].shape[0])]))
                    Rs_idxs.append(torch.LongTensor([rels[j][:, 1], np.arange(rels[j].shape[0])]))
                    Ra = np.zeros((rels[j].shape[0], args.relation_dim)); Ra[:, 0] = 1
                    Ras.append(torch.FloatTensor(Ra))
                    values.append(torch.FloatTensor([1] * rels[j].shape[0]))
                    node_r_idxs.append(node_r_idx[j])
                    node_s_idxs.append(node_s_idx[j])
                    psteps.append(pstep[j])
                    rels_types.append(rels_type[j])

                cnt_clusters += 1

    n_root_level = [0]
    if verbose:
        print("Attr shape (after hierarchy building):", attr.shape)
        print("Object attr:", np.sum(attr, axis=0))
        print("Particle attr:", np.sum(attr[:n_particles], axis=0))
        print("Shape attr:", np.sum(attr[n_particles:n_particles+n_shapes], axis=0))
        print("Roots attr:", np.sum(attr[n_particles+n_shapes:], axis=0))

    ### normalize data
    data = [positions, velocities]
    data_a = normalize(data, stat, var)
    positions, velocities = data[0], data[1]

    if verbose:
        print("Particle positions stats")
        print(positions.shape)
        print(np.min(positions[:n_particles], 0))
        print(np.max(positions[:n_particles], 0))
        print(np.mean(positions[:n_particles], 0))
        print(np.std(positions[:n_particles], 0))

        show_vel_dim = 3
        print("Velocities stats")
        print(velocities.shape)
        print(np.mean(velocities[:n_particles, :show_vel_dim], 0))
        print(np.std(velocities[:n_particles, :show_vel_dim], 0))


    if var:
        state = torch.cat([positions, velocities], 1)
    else:
        state = torch.FloatTensor(np.concatenate([positions, velocities], axis=1))

    if verbose:
        for i in range(count_nodes - 1):
            if np.sum(np.abs(attr[i] - attr[i + 1])) > 1e-6:
                print(i, attr[i], attr[i + 1])

        for i in range(len(Ras)):
            print(i, np.min(node_r_idxs[i]), np.max(node_r_idxs[i]), np.min(node_s_idxs[i]), np.max(node_s_idxs[i]))

    attr = torch.FloatTensor(attr)
    relations = [Rr_idxs, Rs_idxs, values, Ras, node_r_idxs, node_s_idxs, psteps, rels_types]

    return attr, state, relations, n_particles, n_shapes, instance_idx


class FixItDataset(Dataset):

    def __init__(self, args, phase, phases_dict, data_root, verbose):
        self.args = args
        self.phase = phase
        self.phases_dict = phases_dict
        self.verbose = verbose

        if self.args.statf:
            self.stat_path = os.path.join(self.args.data_root, self.args.statf)
        else:
            self.stat_path = None

        self.data_names = ['positions', 'velocities']

        self.training_fpt = self.args.training_fpt
        self.dt = self.training_fpt * self.args.dt
        self.start_timestep = int(1 * self.training_fpt)

        if self.args.n_rollout == None:
            self.all_trials = []
            self.n_rollout = 0

            ddir = os.path.join(data_root, "train_before", self.phase)
            # for ddir in self.data_dir:
            file = open(ddir +  ".txt", "r")
            ddir_root = "/".join(ddir.split("/")[:-1])
            trial_names = [line.strip("\n") for line in file if line != "\n"]
            n_trials = len(trial_names)

            self.all_trials += [os.path.join(ddir_root, trial_name) for trial_name in trial_names]
            self.n_rollout += n_trials

            if phase == "train":
                self.mean_time_step = int(13499/self.n_rollout) + 1
            else:
                self.mean_time_step = 1
        else:
            ratio = self.args.train_valid_ratio
            if phase == 'train':
                self.n_rollout = int(self.args.n_rollout * ratio)
            elif phase == 'valid':
                self.n_rollout = self.args.n_rollout - int(self.args.n_rollout * ratio)
            else:
                raise AssertionError("Unknown phase")

    def __len__(self):
        # each rollout can have different length, sample length in get_item
        return self.n_rollout * self.mean_time_step

    def load_data(self):
        # load the global statistics of "position" and "velocities"

        if self.stat_path is not None:
            self.stat = load_data(self.data_names[:2], self.stat_path)
            for i in range(len(self.stat)):
                self.stat[i] = self.stat[i][-self.args.position_dim:, :]
        else:
            positions_stat = np.zeros((3,3))
            velocities_stat= np.zeros((3,3))

            self.stat = [positions_stat, velocities_stat]

    def __getitem__(self, idx):
        idx = idx % self.n_rollout
        trial_dir = self.all_trials[idx]
        #print(self.args.outf.split("/")[-1], trial_dir)

        data_dir = "/".join(trial_dir.split("/")[:-1])

        trial_fullname = trial_dir.split("/")[-1]

        pkl_path = os.path.join(trial_dir, 'phases_dict.pkl')
        with open(pkl_path, "rb") as f:
            phases_dict = pickle.load(f)
        phases_dict["trial_dir"] = trial_dir

        time_step = phases_dict["time_step"] - self.training_fpt
        idx_timestep = np.random.randint(self.start_timestep, time_step)


        data_path = os.path.join(data_dir, trial_fullname, "arranged_points", str(idx_timestep) + '.npy')
        data_nxt_path = os.path.join(data_dir, trial_fullname, "arranged_points",str(int(idx_timestep + self.training_fpt)) + '.npy')

        data = load_data(self.data_names, data_path)
        data_nxt = load_data(self.data_names, data_nxt_path)

        data_prev_path = os.path.join(data_dir, trial_fullname, "arranged_points",str(max(0, int(idx_timestep - self.training_fpt))) + '.npy')
        data_prev = load_data(self.data_names, data_prev_path)
        _, data, data_nxt = recalculate_velocities([data_prev, data, data_nxt], self.dt, self.data_names)

        attr, state, relations, n_particles, n_shapes, instance_idx = \
                prepare_input(data, self.stat, self.args, phases_dict, self.verbose)

        ### normalized velocities
        data_nxt = normalize(data_nxt, self.stat)
        label = torch.FloatTensor(data_nxt[1][:n_particles])

        return attr, state, relations, n_particles, n_shapes, instance_idx, label, phases_dict

