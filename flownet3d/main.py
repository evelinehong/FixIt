#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import gc
import argparse
from pickletools import ArgumentDescriptor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from data import SceneflowDataset
from model import FlowNet3D
import numpy as np
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from tqdm import tqdm


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)

def scene_flow_EPE_np(pred, labels, mask):
    error = np.sqrt(np.sum((pred - labels)**2, 2) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels*labels, 2) + 1e-20) # B,N
    acc1 = np.sum(np.logical_or((error <= 0.05)*mask, (error/gtflow_len <= 0.05)*mask), axis=1)
    acc2 = np.sum(np.logical_or((error <= 0.1)*mask, (error/gtflow_len <= 0.1)*mask), axis=1)

    mask_sum = np.sum(mask, 1)
    acc1 = acc1[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc1 = np.mean(acc1)
    acc2 = acc2[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc2 = np.mean(acc2)

    EPE = np.sum(error * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    EPE = np.mean(EPE)
    return EPE, acc1, acc2

def test_one_epoch(args, net, test_loader, save=False):
    net.eval()

    total_loss = 0
    total_epe = 0
    total_acc3d = 0
    total_acc3d_2 = 0
    num_examples = 0

    for i, data in tqdm(enumerate(test_loader), total = len(test_loader)):
        pc1, pc2, color1, color2, flow, mask1, filepaths = data
        pc1 = pc1.cuda().transpose(2,1).contiguous().float()
        pc2 = pc2.cuda().transpose(2,1).contiguous().float()
        color1 = color1.cuda().transpose(2,1).contiguous().float()
        color2 = color2.cuda().transpose(2,1).contiguous().float()
        flow = flow.cuda().float()
        mask1 = mask1.cuda().float()

        batch_size = pc1.size(0)
        num_examples += batch_size
        flow_pred = net(pc1, pc2, color1, color2).permute(0,2,1)

        pc1 = pc1 + flow_pred.transpose(1,2)
        # flow_pred2 = net(pc1, pc2, color1, color2).permute(0,2,1)
        # flow_preds2.append(flow_pred2)
        loss = torch.mean(torch.sum((flow_pred - flow) * (flow_pred - flow), -1) / 2.0)
        epe_3d, acc_3d, acc_3d_2 = scene_flow_EPE_np(flow_pred.detach().cpu().numpy(), flow.detach().cpu().numpy(), mask1.detach().cpu().numpy())
        total_epe += epe_3d * batch_size
        total_acc3d += acc_3d * batch_size
        total_acc3d_2 += acc_3d_2 * batch_size
        print('batch EPE 3D: %f\tACC 3D: %f\tACC 3D 2: %f' % (epe_3d, acc_3d, acc_3d_2))

        total_loss += loss.item() * batch_size
        
        if save:
            for (j,filepath) in enumerate(filepaths):
                print (filepath.replace('new', 'flow_pred'))
                with open(filepath.replace('new', 'flow_pred'), "wb") as f:
                    np.save(f, flow_pred[j].cpu().detach().numpy())

        del pc1, pc2, color1, color2, flow, mask1, flow_pred, loss, epe_3d, acc_3d, acc_3d_2
        
    return total_loss * 1.0 / num_examples, total_epe * 1.0 / num_examples, total_acc3d * 1.0 / num_examples, total_acc3d_2 * 1.0 / num_examples
    # return pc1s[0], pc2s[0], flow_preds[0]

def train_one_epoch(args, net, train_loader, opt):
    net.train()
    num_examples = 0
    total_loss = 0
    for i, data in tqdm(enumerate(train_loader), total = len(train_loader)):
        pc1, pc2, color1, color2, flow, mask1, _ = data
        pc1 = pc1.cuda().transpose(2,1).contiguous().float()
        pc2 = pc2.cuda().transpose(2,1).contiguous().float()
        color1 = color1.cuda().transpose(2,1).contiguous().float()
        color2 = color2.cuda().transpose(2,1).contiguous().float()
        flow = flow.cuda().transpose(2,1).contiguous().float()
        mask1 = mask1.cuda().float()

        batch_size = pc1.size(0)
        opt.zero_grad()
        num_examples += batch_size
        flow_pred = net(pc1, pc2, color1, color2)
        loss = torch.mean(torch.sum((flow_pred - flow) ** 2, 1) / 2.0)
        loss.backward()

        opt.step()
        total_loss += loss.item() * batch_size

    return total_loss * 1.0 / num_examples


def test(args, net, test_loader, boardio, textio):
    import json
    test_loss, epe, acc, acc_2 = test_one_epoch(args, net, test_loader, save=True)
    
    textio.cprint('==FINAL TEST==')
    textio.cprint('mean test loss: %f\tEPE 3D: %f\tACC 3D: %f\tACC 3D 2: %f'%(test_loss, epe, acc, acc_2))
    del test_loss, epe, acc, acc_2

def train(args, net, train_loader, test_loader, boardio, textio):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    # scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)
    scheduler = StepLR(opt, 10, gamma = 0.7)

    best_test_loss = np.inf

    for epoch in range(args.epochs):
        textio.cprint('==epoch: %d, learning rate: %f=='%(epoch, opt.param_groups[0]['lr']))
        train_loss = train_one_epoch(args, net, train_loader, opt)
        textio.cprint('mean train EPE loss: %f'%train_loss)

        test_loss, epe, acc, acc_2 = test_one_epoch(args, net, test_loader)
        textio.cprint('mean test loss: %f\tEPE 3D: %f\tACC 3D: %f\tACC 3D 2: %f'%(test_loss, epe, acc, acc_2))
        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            textio.cprint('best test loss till now: %f'%test_loss)
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
        
        scheduler.step()

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--category', type=str, default='fridge', metavar='N',
                        help='Name of the category')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='flownet', metavar='N',
                        choices=['flownet'],
                        help='Model to use, [flownet]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Point Number [default: 2048]')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Whether to test on unseen category')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')

    args = parser.parse_args()

    # CUDA settings
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    boardio = []
    _init_(args)

    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

    train_loader = DataLoader(
        SceneflowDataset(npoints=args.num_points, cat = args.category, partition='train', sample=True),
        batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        SceneflowDataset(npoints=args.num_points, cat = args.category, partition='val', sample=True),
        batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(
        SceneflowDataset(npoints=args.num_points, cat = args.category, partition='test'),
        batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    if args.model == 'flownet':
        net = FlowNet3D(args).cuda()
        net.apply(weights_init)
        model_path = './pretrained_model/model.best.t7'
        net.load_state_dict(torch.load(model_path), strict=False)
        if args.eval:
            if args.model_path is '':
                model_path = '../checkpoints/%s/flownet/model.best.t7'%args.category
            else:
                model_path = args.model_path
                print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            net.load_state_dict(torch.load(model_path), strict=False)
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        raise Exception('Not implemented')

    if args.eval:
        train_loader = DataLoader(
        SceneflowDataset(npoints=args.num_points, cat = args.category, partition='train'),
        batch_size=args.batch_size, shuffle=True, drop_last=True)

        test(args, net, test_loader, boardio, textio)
        test(args, net, train_loader, boardio, textio)
    else:
        train(args, net, train_loader, val_loader, boardio, textio)

    print('FINISH')

if __name__ == '__main__':
    main()