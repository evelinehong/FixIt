import os
import random
import argparse
import socket
import getpass
import yaml
import re
import os

def mkdir(path, is_assert=False):
    if is_assert:
        assert(not os.path.exists(path)), f"{path} exists, delete it first if you want to overwrite"
    if not os.path.exists(path):
        os.makedirs(path)

def get_source_dir():
    return os.getcwd()

def get_query_dir(query_dir):
    hostname = socket.gethostname()
    username = getpass.getuser()
    paths_yaml_fn = os.path.join(get_source_dir(), 'paths.yaml')
    with open(paths_yaml_fn, 'r') as f:
        paths_config = yaml.load(f, Loader=yaml.Loader)

    for hostname_re in paths_config:
        if re.compile(hostname_re).match(hostname) is not None:
            for username_re in paths_config[hostname_re]:
                if re.compile(username_re).match(username) is not None:
                    return paths_config[hostname_re][username_re][query_dir]

    raise Exception('No matching hostname or username in config file')

parser = argparse.ArgumentParser(description='Point Cloud Registration')
parser.add_argument('--category', type=str, default='fridge', metavar='N',
                    help='Name of the category')

args = parser.parse_args()

data_path = os.path.join("../data/%s/shapes"%args.category)

train_valid_ratio = 0.95

data_root = os.path.join(data_path, "train_before")

scene_path = os.path.join(data_root)

trial_names = [file for file in os.listdir(scene_path) if not file.endswith(".txt") and not file.endswith(".swp")]

random.shuffle(trial_names)

f = open(os.path.join(scene_path, "train.txt"), "w")
total_trials = len(trial_names)
n_trains = int(total_trials * train_valid_ratio)
for trial_name in trial_names[:n_trains]:
    f.write(trial_name + "\n")

f2 = open(os.path.join(scene_path, "valid.txt"), "w")
for trial_name in trial_names[n_trains:]:
    f2.write(trial_name + "\n")

print("train_before", total_trials, "train, valid:", n_trains, ",", total_trials - n_trains)




