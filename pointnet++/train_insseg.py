
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
from scipy.optimize import linear_sum_assignment

from pathlib import Path
from tqdm import tqdm
from data_utils.segmentation_loader import PartNormalDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
torch.backends.cudnn.enabled = False

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(4)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_ins_seg_msg', help='model name')
    parser.add_argument('--cat', type=str, default='fridge', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')

    return parser.parse_args()

def hungarian_matching(pred_x, gt_x, curnmasks):
    """ pred_x, gt_x: B x nmask x n_point
        curnmasks: B
        return matching_idx: B x nmask x 2 """
    batch_size = gt_x.shape[0]
    nmask = gt_x.shape[1]
    matching_score = np.matmul(gt_x, np.transpose(pred_x, axes=[0, 2, 1])) # B x nmask x nmask
    matching_score = 1 - np.divide(matching_score, np.maximum(np.expand_dims(np.sum(pred_x, 2), 1)+np.sum(gt_x, 2, keepdims=True) - matching_score, 1e-8))
    matching_idx = np.zeros((batch_size, nmask, 2)).astype('int32')
    curnmasks = curnmasks.astype('int32')
    for i, curnmask in enumerate(curnmasks):
        row_ind, col_ind = linear_sum_assignment(matching_score[i, :curnmask, :])
        matching_idx[i, :curnmask, 0] = row_ind
        matching_idx[i, :curnmask, 1] = col_ind
    return matching_idx

def gather_nd(params, indices):
    """ The same as tf.gather_nd but batched gather is not supported yet.
    indices is an k-dimensional integer tensor, best thought of as a (k-1)-dimensional tensor of indices into params, where each element defines a slice of params:

    output[\\(i_0, ..., i_{k-2}\\)] = params[indices[\\(i_0, ..., i_{k-2}\\)]]

    Args:
        params (Tensor): "n" dimensions. shape: [x_0, x_1, x_2, ..., x_{n-1}]
        indices (Tensor): "k" dimensions. shape: [y_0,y_2,...,y_{k-2}, m]. m <= n.

    Returns: gathered Tensor.
        shape [y_0,y_2,...y_{k-2}] + params.shape[m:] 

    """
    orig_shape = list(indices.shape)
    num_samples = np.prod(orig_shape[:-1])
    m = orig_shape[-1]
    n = len(params.shape)

    if m <= n:
        out_shape = orig_shape[:-1] + list(params.shape)[m:]
    else:
        raise ValueError(
            f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
        )

    indices = indices.reshape((num_samples, m)).transpose(0, 1).tolist()
    output = params[indices]    # (num_samples, ...)
    return output.reshape(out_shape).contiguous()

# copy from https://github.com/ericyi/articulated-part-induction/blob/master/model.py
def iou(pred_x, gt_x, valid, n_point, nmask, end_points):
    pred_x = pred_x.transpose(2,1)
    gt_x = gt_x.transpose(2,1)

    matching_idx = hungarian_matching (pred_x.detach().cpu().numpy(), gt_x.detach().cpu().numpy(), torch.sum(valid, -1).detach().cpu().numpy())
    end_points['matching_idx'] = matching_idx

    matching_idx_row = torch.tensor(matching_idx[:, :, 0]).long()
    
    idx = torch.where((matching_idx_row >= 0))
    idx = torch.stack(idx, dim=1)
    # print (matching_idx_row.reshape(-1,1).size())
    # print (torch.unsqueeze(idx[:, 0], -1).size())
    matching_idx_row = torch.cat((torch.unsqueeze(idx[:, 0], -1), matching_idx_row.reshape(-1,1)), 1)
    gt_x_matched = gather_nd(gt_x, matching_idx_row).reshape(-1, nmask, n_point)

    matching_idx_column = torch.tensor(matching_idx[:, :, 1]).long()
    idx = torch.where(matching_idx_column >= 0)
    idx = torch.stack(idx, dim=1)
    matching_idx_column = torch.cat((torch.unsqueeze(idx[:, 0], -1), matching_idx_column.reshape(-1,1)), 1)
    pred_x_matched = gather_nd(pred_x, matching_idx_column).reshape(-1, nmask, n_point)
    # compute meaniou
    matching_score = torch.sum(gt_x_matched * pred_x_matched, 2)
    iou_all = matching_score / (torch.sum(gt_x_matched, 2) + torch.sum(pred_x_matched, 2) - matching_score + 1e-8)
    end_points['per_shape_all_iou'] = iou_all

    meaniou = torch.sum(iou_all * valid, 1) / (torch.sum(valid, -1) + 1e-8) # B
    return meaniou, end_points

def get_ins_loss(mask_pred, mask_gt, valid, end_points):
    """ Input:  mask_pred   B x K x N
                mask_gt     B x K x N
                gt_valid    B x K
    """
    num_ins = mask_pred.size()[2]
    num_point = mask_pred.size()[1]
    meaniou, end_points = iou(mask_pred, mask_gt, valid, num_point, num_ins, end_points)
    end_points['per_shape_mean_iou'] = meaniou
    loss = - torch.mean(meaniou)
    return loss, end_points

def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.
    this function don't contain except handle code. so use this carefully
    when indice repeats, don't support repeat add which is supported
    in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret 

def get_conf_loss(conf_pred, gt_valid, end_points):
    """ Input:  conf_pred       B x K
                gt_valid        B x K
    """
    batch_size = conf_pred.size()[0]
    nmask = conf_pred.size()[1]

    iou_all = end_points['per_shape_all_iou']
    matching_idx = end_points['matching_idx']

    matching_idx_column = matching_idx[:, :, 1]
    idx = torch(matching_idx_column >= 0)
    idx = torch.stack(idx, dim=1)
    all_indices = torch.cat((torch.unsqueeze(idx[:, 0], -1), matching_idx_column.reshape(-1, 1)), 1)
    all_indices = all_indices.reshape(batch_size, nmask, 2)

    valid_idx = torch.where(gt_valid >= 0.5)
    predicted_indices = gather_nd(all_indices, valid_idx)
    valid_iou = gather_nd(iou_all, valid_idx)

    conf_target = scatter_nd(predicted_indices, valid_iou, torch.tensor([batch_size, nmask]))
    end_points['per_part_conf_target'] = conf_target

    per_part_loss = (conf_pred - conf_target) ** 2
    end_points['per_part_loss'] = per_part_loss

    target_pos_mask = conf_target > 0.1
    target_neg_mask = 1.0 - target_pos_mask
    
    pos_per_shape_loss = torch.sum(target_pos_mask * per_part_loss, axis=-1) / torch.maximum(1e-6, torch.sum(target_pos_mask, axis=-1))
    neg_per_shape_loss = torch.sum(target_neg_mask * per_part_loss, axis=-1) / torch.maximum(1e-6, torch.sum(target_neg_mask, axis=-1))

    per_shape_loss = pos_per_shape_loss + neg_per_shape_loss
    end_points['per_shape_loss'] = per_shape_loss

    loss = torch.mean(per_shape_loss)
    return loss, end_points

def get_l21_norm(mask_pred, end_points):
    """ Input:  mask_pred           B x K x N
                other_mask_pred     B x N
    """
    
    num_point = mask_pred.size()[1]
    full_mask = mask_pred + 1e-6
    per_shape_l21_norm = torch.sum((torch.norm(full_mask, dim=-1)), dim=-1)
    end_points['per_shape_l21_norm'] = per_shape_l21_norm / num_point

    loss = torch.mean(per_shape_l21_norm)
    return loss, end_points

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('ins_seg')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    TRAIN_DATASET = PartNormalDataset(cat=args.cat, npoints=args.npoint, split='train', sample=True)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    TEST_DATASET = PartNormalDataset(cat=args.cat, npoints=args.npoint, split='val', sample=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_classes = 4
    num_part = 4

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    classifier = MODEL.get_model(num_part, normal_channel=True).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    start_epoch = 0
    classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        average_iou = []
        losses = []
        for i, (points, label, target, target2, valid, _) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target, valid = points.float().cuda(), label.long().cuda(), target.long().cuda(), valid.cuda()
            points = points.transpose(2, 1)
            target2 = target2.long().cuda()
            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
            loss1, end_points = get_ins_loss(seg_pred, target, valid, end_points = dict())
            loss2, end_points = get_l21_norm(seg_pred, end_points)           

            loss1 *= 100
            loss2 *= 0.01

            
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target2 = target2.view(-1, 1)[:, 0]
            target = target.contiguous().view(-1, num_part)
            
            pred_choice = seg_pred.data.max(1)[1]
            target = target.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))
            loss3 = criterion(seg_pred, target2, trans_feat)
            loss = loss1+loss2
            # loss = criterion(seg_pred, target, trans_feat)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

        loss = np.mean(losses)
        # log_string('Train Average IOU: %.5f' % average_iou)
        log_string('Train Loss: %.5f' % loss)

        with torch.no_grad():
            test_metrics = {}
            best_loss = 10000
            classifier = classifier.eval()
            average_iou = []
            losses = []
            mean_correct = []
            for batch_id, (points, label, target, target2, valid, _) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target, valid = points.float().cuda(), label.long().cuda(), target.long().cuda(), valid.cuda()
                points = points.transpose(2, 1)
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                loss, end_points = get_ins_loss(seg_pred, target, valid, end_points = dict())
                loss *= 10
                # average_iou.append(end_points['per_shape_all_iou'])
                losses.append(loss.item())

                seg_pred = seg_pred.contiguous().view(-1, num_part)
                target = target.contiguous().view(-1, num_part)
                
                pred_choice = seg_pred.data.max(1)[1]
                target = target.data.max(1)[1]

                correct = pred_choice.eq(target.data).cpu().sum()
                mean_correct.append(correct.item() / (args.batch_size * args.npoint))

            # average_iou = np.mean(average_iou)
            loss = np.mean(losses)
            # log_string('Train Average IOU: %.5f' % average_iou)
            log_string('Test Loss: %.5f' % loss)

            if (loss < best_loss):
                loss = best_loss
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')

if __name__ == '__main__':
    args = parse_args()
    main(args)
