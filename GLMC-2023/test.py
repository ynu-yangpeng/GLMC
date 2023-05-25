import sys
import os
import argparse
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])
os.path.dirname
os.path.abspath(os.path.dirname(__file__))
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
from model import Resnet_LT
from imbalance_data import cifar10Imbanlance,cifar100Imbanlance,dataset_lt_data


def eval_training(model, val_loader, args):
    size = len(val_loader)
    model.eval()
    correct_top1 = []
    for i, (inputs, labels) in enumerate(val_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            logits = model(inputs, train=False)
            top1, _ = accuracy(logits.data, labels.data, topk=(1, 5))
            correct_top1.append(top1.cpu().numpy())
        output = 'Test:  ' + str(i) + ' Prec@1:  ' + str(top1.item())
        print(output)
    correct_top1 = sum(correct_top1) / len(correct_top1)
    return correct_top1

def get_model(args):
    if args.dataset == "ImageNet-LT" or args.dataset == "iNaturelist2018":
        net = Resnet_LT.resnext50_32x4d(num_classes=args.num_classes)
        print("=> creating model '{}'".format('resnext50_32x4d'))
    else:
        if args.arch == 'resnet50':
            net = ResNet_cifar.resnet50(num_class=args.num_classes)
        elif args.arch == 'resnet18':
            net = ResNet_cifar.resnet18(num_class=args.num_classes)
        elif args.arch == 'resnet34':
            net = ResNet_cifar.resnet34(num_class=args.num_classes)
        print("=> creating model '{}'".format(args.arch))
    return net

def get_dataset(args):
    _,transform_val = util.get_transform(args.dataset)
    if args.dataset == 'cifar10':
        testset = cifar10Imbanlance.Cifar10Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False, transform=transform_val,file_path=os.path.join(args.root,'cifar-10-batches-py/'))
        print("load cifar10")
        return testset

    if args.dataset == 'cifar100':
        testset = cifar100Imbanlance.Cifar100Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False, transform=transform_val,file_path=os.path.join(args.root,'cifar-100-python/'))
        print("load cifar100")
        return testset

    if args.dataset == 'ImageNet-LT':
        testset = dataset_lt_data.LT_Dataset(args.root, args.dir_test_txt, transform_val)
        return testset

    if args.dataset == 'iNaturelist2018':
        testset = dataset_lt_data.LT_Dataset(args.root, args.dir_test_txt,transform_val)
        return testset

def main():
    args = parser.parse_args()
    global train_cls_num_list
    global cls_num_list_cuda

    if args.gpu is not None:
        print("Use GPU: {} for testing".format(args.gpu))
    # create model
    num_classes = args.num_classes
    model = get_model(args)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # test from a checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cuda:0')
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.resume))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    val_dataset = get_dataset(args)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers, pin_memory=True)

    cls_num_list = [0] * num_classes
    for label in val_dataset.targets:
        cls_num_list[label] += 1
    train_cls_num_list = np.array(cls_num_list)
    cls_num_list_cuda = torch.from_numpy(np.array(cls_num_list)).float().cuda()

    print("Testing started!")
    # switch to evaluate mode
    model.eval()
    acc = eval_training(model, val_loader, args)
    print("All acc : "+str(acc))

if __name__ == '__main__':
    # test set
    parser = argparse.ArgumentParser(description="Global and Local Mixture Consistency Cumulative Learning")
    parser.add_argument('--dataset', type=str, default='cifar100',help="cifar10,cifar100,ImageNet-LT,iNaturelist2018")
    parser.add_argument('--root', type=str, default='/data/',help="dataset setting")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',choices=('resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d'))
    parser.add_argument('--imbanlance_rate', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes ',choices=('10', '100', '1000', '8142'))
    parser.add_argument('-b', '--batch_size', default=128, type=int,metavar='N', help='mini-batch size')
    # etc.
    parser.add_argument('-p', '--print_freq', default=100, type=int, metavar='N',help='print frequency (default: 100)')
    parser.add_argument('--gpu', default=None, type=int,help='GPU id to use.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default='model path', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('--root_model', type=str, default='GLMC-CVPR2023/output/')
    parser.add_argument('--store_name', type=str, default='GLMC-CVPR2023/output/')
    parser.add_argument('--dir_train_txt', type=str,default="GLMC-CVPR2023/data/data_txt/iNaturalist18_train.txt")
    parser.add_argument('--dir_test_txt', type=str,default="GLMC-CVPR2023/data/data_txt/iNaturalist18_val.txt")
    main()
