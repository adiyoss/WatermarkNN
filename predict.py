from __future__ import print_function

import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable

from helpers.loaders import *
from helpers.utils import progress_bar

parser = argparse.ArgumentParser(description='Test models on CIFAR-10 and watermark sets.')
parser.add_argument('--model_path', default='checkpoint/teacher-cifar100-2.t7', help='the model path')
parser.add_argument('--wm_path', default='./data/trigger_set/', help='the path the wm set')
parser.add_argument('--wm_lbl', default='labels-cifar.txt', help='the path the wm random labels')
parser.add_argument('--testwm', action='store_true', help='test the wm set or cifar10 dataset.')
parser.add_argument('--db_path', default='./data', help='the path to the root folder of the test data')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 100

# Data
if args.testwm:
    print('Loading watermark images')
    loader = getwmloader(args.wm_path, batch_size, args.wm_lbl)
else:
    _, loader, _ = getdataloader('cifar10', args.db_path, args.db_path, batch_size)

# Model
# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.exists(args.model_path), 'Error: no checkpoint found!'
checkpoint = torch.load(args.model_path)
net = checkpoint['net']
acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()

net.eval()
test_loss = 0
correct = 0
total = 0
for batch_idx, (inputs, targets) in enumerate(loader):
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = net(inputs)
    loss = criterion(outputs, targets)

    test_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()

    progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                 % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
