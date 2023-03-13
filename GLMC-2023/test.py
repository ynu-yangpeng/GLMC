import sys
import os
import time
import argparse
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])
os.path.dirname #获取当前运行脚本的绝对路径
os.path.abspath(os.path.dirname(__file__)) #获取当前脚本的父路径的绝对路径
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])
import torch
import torch.nn as nn
import numpy as np
import random
from torch.backends import cudnn
import torch.nn.functional as F
from utils import util
from utils.util import *
from model import ResNet_cifar
from model import Resnet_ImageNet_LT
from imbalance_data import cifar10Imbanlance,cifar100Imbanlance,dataset_lt_data
import logging
import datetime
import math
from sklearn.metrics import confusion_matrix
from loss import *
import warnings

best_acc1 = 0
def eval_training(model,val_loader,args):
        model.eval()
        correct_top1 = []
        predList = np.array([])
        grndList = np.array([])
        for i,  (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            with torch.no_grad():
                logits = model(inputs,train=False)
                softmaxScores = F.softmax(logits, dim=1)
                predLabels = softmaxScores.argmax(dim=1).detach().squeeze()
                predList = np.concatenate((predList, predLabels.cpu().numpy()))
                grndList = np.concatenate((grndList, labels.cpu().numpy()))
                top1, top5 = accuracy(logits.data, labels.data, topk=(1,5))
                correct_top1.append(top1.cpu().numpy())
            output = 'Test:  ' + str(i) +' Prec@1:  ' + str(top1.item())
            print(output)

        correct_top1 = sum(correct_top1) / len(correct_top1)
        return correct_top1

def get_model(args):
    if args.dataset == "ImageNet-LT":
        print("=> creating model '{}'".format('resnext50_32x4d'))
        net = Resnet_ImageNet_LT.resnext50_32x4d(num_classes=args.num_classes)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch == 'resnet50':
            net = ResNet_cifar.resnet50(num_class=args.num_classes)
        elif args.arch == 'resnet18':
            net = ResNet_cifar.resnet18(num_class=args.num_classes)
        elif args.arch == 'resnet34':
            net = ResNet_cifar.resnet34(num_class=args.num_classes)
    return net

def get_dataset(args):
    transform_train,transform_val = util.get_transform(args.dataset)
    if args.dataset == 'cifar10':
        trainset = cifar10Imbanlance.Cifar10Imbanlance(transform=util.TwoCropTransform(transform_train),imbanlance_rate=args.imbanlance_rate, train=True,file_path=args.root)
        testset = cifar10Imbanlance.Cifar10Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False, transform=transform_val,file_path=args.root)
        print("load cifar10")
        return trainset,testset

    if args.dataset == 'cifar100':
        trainset = cifar100Imbanlance.Cifar100Imbanlance(transform=util.TwoCropTransform(transform_train),imbanlance_rate=args.imbanlance_rate, train=True,file_path=os.path.join(args.root,'cifar-100-python/'))
        testset = cifar100Imbanlance.Cifar100Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False, transform=transform_val,file_path=args.root)
        print("load cifar100")
        return trainset,testset

    if args.dataset == 'ImageNet-LT':
        trainset = dataset_lt_data.LT_Dataset(args.root, args.dir_train_txt, util.TwoCropTransform(transform_train))
        testset = dataset_lt_data.LT_Dataset(args.root, args.dir_test_txt, transform_val)
        return trainset,testset

def main():
    args = parser.parse_args()
    global best_acc1
    global train_cls_num_list
    global cls_num_list_cuda

    if args.gpu is not None:
        print("Use GPU: {} for testing".format(args.gpu))
    # create model
    num_classes = args.num_classes
    model = get_model(args)

    # test from a checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cuda:0')
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(args.gpu)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.resume))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # Data loading code
    _,val_dataset = get_dataset(args)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers, pin_memory=True)

    cls_num_list = [0] * num_classes
    for label in val_dataset.targets:
        cls_num_list[label] += 1
    train_cls_num_list = np.array(cls_num_list)
    cls_num_list_cuda = torch.from_numpy(np.array(cls_num_list)).float().cuda()

    start_time = time.time()
    print("Testing started!")

    flag = 'val'

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    acc = eval_training(model, val_loader, args)
    print(acc)

if __name__ == '__main__':
    # test set
    parser = argparse.ArgumentParser(description="Global and Local Mixture Consistency Cumulative Learning")
    parser.add_argument('--dataset', type=str, default='ImageNet-LT',help="cifar10,cifar100,ImageNet-LT")
    parser.add_argument('--root', type=str, default='/root/NFS_Data/Public_DataSet/',help="dataset setting")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34')
    parser.add_argument('--num_classes', default=1000, type=int, help='number of classes ')
    parser.add_argument('-b', '--batch_size', default=128, type=int,metavar='N', help='mini-batch size')
    # etc.
    parser.add_argument('-p', '--print_freq', default=100, type=int, metavar='N',help='print frequency (default: 100)')
    parser.add_argument('--gpu', default=None, type=int,help='GPU id to use.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default='/root/PaperProject/pretrain/57.20%.tar', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('--root_model', type=str, default='/root/PaperProject/GLMC-CVPR2023/output/')
    parser.add_argument('--store_name', type=str, default='/root/PaperProject/GLMC-CVPR2023/output/')
    parser.add_argument('--dir_train_txt', type=str,default="/root/PaperProject/GLMC-CVPR2023/data/data_txt/ImageNet_LT_train.txt")
    parser.add_argument('--dir_test_txt', type=str,default="/root/PaperProject/GLMC-CVPR2023/data/data_txt/ImageNet_LT_test.txt")
    main()
