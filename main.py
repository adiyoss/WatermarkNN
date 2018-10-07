from __future__ import print_function

import argparse
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from helpers.loaders import *
from helpers.utils import adjust_learning_rate
from models import ResNet18
from trainer import test, train

parser = argparse.ArgumentParser(description='Train CIFAR-10 models with watermaks.')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--train_db_path', default='./data', help='the path to the root folder of the traininng data')
parser.add_argument('--test_db_path', default='./data', help='the path to the root folder of the traininng data')
parser.add_argument('--dataset', default='cifar10', help='the dataset to train on [cifar10]')
parser.add_argument('--wm_path', default='./data/trigget_set/', help='the path the wm set')
parser.add_argument('--wm_lbl', default='labels-cifar.txt', help='the path the wm random labels')
parser.add_argument('--batch_size', default=100, type=int, help='the batch size')
parser.add_argument('--wm_batch_size', default=2, type=int, help='the wm batch size')
parser.add_argument('--max_epochs', default=60, type=int, help='the maximum number of epochs')
parser.add_argument('--lradj', default=20, type=int, help='multiple the lr by 0.1 every n epochs')
parser.add_argument('--save_dir', default='./checkpoint/', help='the path to the model dir')
parser.add_argument('--save_model', default='model.t7', help='model name')
parser.add_argument('--load_path', default='./checkpoint/ckpt.t7', help='the path to the pre-trained model, to be used with resume flag')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--wmtrain', '-wmt', action='store_true', help='train with wms?')
parser.add_argument('--log_dir', default='./log', help='the path the log dir')
parser.add_argument('--runname', default='train', help='the exp name')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

LOG_DIR = args.log_dir
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
logfile = os.path.join(LOG_DIR, 'log_' + str(args.runname) + '.txt')
confgfile = os.path.join(LOG_DIR, 'conf_' + str(args.runname) + '.txt')

# save configuration parameters
with open(confgfile, 'w') as f:
    for arg in vars(args):
        f.write('{}: {}\n'.format(arg, getattr(args, arg)))

trainloader, testloader, n_classes = getdataloader(
    args.dataset, args.train_db_path, args.test_db_path, args.batch_size)

wmloader = None
if args.wmtrain:
    print('Loading watermark images')
    wmloader = getwmloader(args.wm_path, args.wm_batch_size, args.wm_lbl)

# create the model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.exists(args.load_path), 'Error: no checkpoint found!'
    checkpoint = torch.load(args.load_path)
    net = checkpoint['net']
    acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    net = ResNet18(num_classes=n_classes)

net = net.to(device)
# support cuda
if device == 'cuda':
    print('Using CUDA')
    print('Parallel training on {0} GPUs.'.format(torch.cuda.device_count()))
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# loading wm examples
if args.wmtrain:
    print("WM acc:")
    test(net, criterion, logfile, wmloader, device)

# start training
for epoch in range(start_epoch, start_epoch + args.max_epochs):
    # adjust learning rate
    adjust_learning_rate(args.lr, optimizer, epoch, args.lradj)

    train(epoch, net, criterion, optimizer, logfile,
          trainloader, device, wmloader)

    print("Test acc:")
    acc = test(net, criterion, logfile, testloader, device)

    if args.wmtrain:
        print("WM acc:")
        test(net, criterion, logfile, wmloader, device)

    print('Saving..')
    state = {
        'net': net.module if device is 'cuda' else net,
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    torch.save(state, os.path.join(args.save_dir, args.save_model))
