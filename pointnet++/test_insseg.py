
import argparse
import os
from data_utils.segmentation_loader import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(4)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    parser.add_argument('--cat', type=str, default='fridge', help='model name')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/ins_seg/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    TEST_DATASET = PartNormalDataset(cat=args.cat, npoints=args.num_point, split='test')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 4
    num_part = 4

    '''MODEL LOADING'''

    model_name = 'pointnet2_ins_seg_msg'
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=True).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')

    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        test_metrics = {}

        classifier = classifier.eval()
        for batch_id, (points, label, target, target2, valid, datapaths) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                      smoothing=0.9):
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points2 = points.detach().cpu().numpy().tolist()
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()

            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred

            seg_pred = vote_pool / args.num_votes
            seg_pred = seg_pred.contiguous().view(-1, num_part)

            pred_choice = seg_pred.data.max(1)[1]
            pred_choice = pred_choice.view(-1, 2048)
            
            for j, pred in enumerate(pred_choice):
                datapath = datapaths[j]
                with open(os.path.join(datapath, "instance_segmentation.npy"), "wb") as file:
                    np.save(file, pred.detach().cpu().numpy())

    TEST_DATASET = PartNormalDataset(cat=args.cat, npoints=args.num_point, split='train')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    with torch.no_grad():
        test_metrics = {}

        classifier = classifier.eval()
        for batch_id, (points, label, target, target2, valid, datapaths) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                      smoothing=0.9):
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points2 = points.detach().cpu().numpy().tolist()
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()

            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred

            seg_pred = vote_pool / args.num_votes
            seg_pred = seg_pred.contiguous().view(-1, num_part)

            pred_choice = seg_pred.data.max(1)[1]
            pred_choice = pred_choice.view(-1, 2048)
            
            for j, pred in enumerate(pred_choice):
                datapath = datapaths[j]
                with open(os.path.join(datapath, "instance_segmentation.npy"), "wb") as file:
                    np.save(file, pred.detach().cpu().numpy())

if __name__ == '__main__':
    args = parse_args()
    main(args)
