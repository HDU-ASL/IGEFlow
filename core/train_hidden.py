from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raft_hidden import RAFT
import evaluate_hidden
import datasets

from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision.utils import flow_to_image

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, lamda=0, LD=0, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    LD = LD.mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        'flow_loss': flow_loss.item(),
        'LD': LD.item()
    }

    return flow_loss + LD * lamda, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


def save_tensor2png(tensor_data, path):
        # 将 Tensor 转移到 CPU，并转换为 NumPy 数组
    tensor_data = tensor_data.cpu().numpy()

    # 确保 Tensor 数据范围在 [0, 255] 之间
    tensor_data = np.clip(tensor_data, 0, 255).astype(np.uint8)

    # 转换为 PIL 图像
    image = Image.fromarray(tensor_data.transpose(1, 2, 0))  # 转换为 HWC 形式

    # 保存为 PNG 文件
    image.save(path)

def save_flow2png(flow_tensor, path):
    # 使用 flow_to_image 函数转换为图像
    image_tensor = flow_to_image(flow_tensor)

    save_tensor2png(image_tensor, path)
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self, logdir):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter(logdir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics, logdir):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status(logdir)
            self.running_loss = {}

    def write_dict(self, results, logdir):
        if self.writer is None:
            self.writer = SummaryWriter(logdir)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
        print('load restore_ckpt: ', args.restore_ckpt)

    model.module.feature_load_and_freeze()

    print("Parameter Count: %d" % count_parameters(model))
    model.cuda()
    model.train()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 5000
    add_noise = False

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, image1_H, image2_H, flow, valid = [x.cuda() for x in data_blob]

            # save_tensor2png(image1[0],'runs/debug/1.png')
            # save_tensor2png(image2[0],'runs/debug/2.png')
            # save_tensor2png(image1_H[0],'runs/debug/1_h.png')
            # save_tensor2png(image2_H[0],'runs/debug/2_h.png')
            # save_tensor2png(flow_to_image(flow[0]),'runs/debug/3.png')

            flow_predictions, LD = model(image1, image2, image1_H, image2_H, iters=args.iters)            

            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma, args.lamda, LD)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics, f'runs/{args.name}/')

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'runs/%s/checkpoints/%d_%s.pth' % (args.name, total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate_hidden.validate_Canon(model.module))
                        results.update(evaluate_hidden.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate_hidden.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate_hidden.validate_kitti(model.module))

                logger.write_dict(results, f'runs/{args.name}/')
                
                model.train()
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'runs/%s/checkpoints/%s.pth' % (args.name, args.name)
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--feature_guide_ckpt', help="restore checkpoint") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--extractor_denoise', action='store_true', help='use extractor_denoise model')
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--lamda', type=float, default=0)
    parser.add_argument('--use_enhance', action='store_true', default=False)


    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir(f'runs/{args.name}/checkpoints'):
        os.makedirs(f'runs/{args.name}/checkpoints', exist_ok=True)
    
    train(args)