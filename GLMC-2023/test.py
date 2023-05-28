import os
import argparse
import torch
from utils import util
from utils.util import *
from model import ResNet_cifar
from model import Resnet_LT
from imbalance_data import cifar10Imbanlance,cifar100Imbanlance,dataset_lt_data


def validate(model,val_loader,args):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            # compute output
            output = model(input, train=False)
            # measure accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))
            output = 'Testing:  ' + str(i) + ' Prec@1:  ' + str(top1.val) + ' Prec@5:  ' + str(top5.val)
            print(output, end="\r")
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(flag='val', top1=top1, top5=top5))
        print(output)

def get_model(args):
    if args.dataset == "ImageNet-LT" or args.dataset == "iNaturelist2018":
        net = Resnet_LT.resnext50_32x4d(num_classes=args.num_classes)
        print("=> creating model '{}'".format('resnext50_32x4d'))
    else:
        if args.arch == 'resnet50':
            net = ResNet_cifar.resnet50(num_class=args.num_classes)
        elif args.arch == 'resnet18':
            net = ResNet_cifar.resnet18(num_class=args.num_classes)
        elif args.arch == 'resnet32':
            net = ResNet_cifar.resnet32(num_class=args.num_classes)
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
        print("load ImageNet-LT")
        return testset

    if args.dataset == 'iNaturelist2018':
        testset = dataset_lt_data.LT_Dataset(args.root, args.dir_test_txt,transform_val)
        print("load iNaturelist2018")
        return testset

def main():
    args = parser.parse_args()

    if args.gpu is not None:
        print("Use GPU: {} for testing".format(args.gpu))
    # create model
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


    print("Testing started!")
    # switch to evaluate mode
    model.eval()
    validate(model, val_loader, args)

if __name__ == '__main__':
    # test set
    parser = argparse.ArgumentParser(description="Global and Local Mixture Consistency Cumulative Learning")
    parser.add_argument('--dataset', type=str, default='cifar100',help="cifar10,cifar100,ImageNet-LT,iNaturelist2018")
    parser.add_argument('--root', type=str, default='/data/',help="dataset setting")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',choices=('resnet18', 'resnet32', 'resnet50', 'resnext50_32x4d'))
    parser.add_argument('--imbanlance_rate', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes ',choices=('10', '100', '1000', '8142'))
    parser.add_argument('-b', '--batch_size', default=100, type=int,metavar='N', help='mini-batch size')
    # etc.
    parser.add_argument('-p', '--print_freq', default=100, type=int, metavar='N',help='print frequency (default: 100)')
    parser.add_argument('--gpu', default=None, type=int,help='GPU id to use.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default='model path', type=str, metavar='PATH',help='path to latest checkpoint')
    parser.add_argument('--root_model', type=str, default='GLMC-CVPR2023/output/')
    parser.add_argument('--store_name', type=str, default='GLMC-CVPR2023/output/')
    parser.add_argument('--dir_train_txt', type=str,default="GLMC-CVPR2023/data/data_txt/iNaturalist18_train.txt")
    parser.add_argument('--dir_test_txt', type=str,default="GLMC-CVPR2023/data/data_txt/iNaturalist18_val.txt")
    main()
