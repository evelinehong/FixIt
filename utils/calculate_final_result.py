import os
import json
import argparse

all = 0
hit = 0

parser = argparse.ArgumentParser(description='Point Cloud Registration')
parser.add_argument('--category', type=str, default='fridge', metavar='N',
                    help='Name of the category')
args = parser.parse_args()

root = os.path.join("../data/%s/shapes"%args.category, "%s_before"%args.split)
root2 = os.path.join("../data/%s/shapes"%args.category, args.split)

for shape in os.listdir(root):
    max_acc = -10000
    all += 1
    for i in range(5):
        answer = json.load(open(os.path.join(root2, shape+"_%d"%i, "final_pred_0.json")))[1]

        if answer > max_acc:
            pred = i
            max_acc = answer

        correct = json.load(open(os.path.join(root2, shape+"_%d"%i, "answer.json")))['correct']
        if correct:
            gt = i

    if gt == pred:
        hit += 1

print (hit, all)   
print (hit/all)