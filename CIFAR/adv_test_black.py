from comet_ml import Experiment

import numpy as np
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.allconv import AllConvNet
from models.wrn import WideResNet
from models.distnet import DistanceNet
from skimage.filters import gaussian as gblur
from PIL import Image as PILImage

from torch.utils.data import DataLoader, TensorDataset

# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std
    import utils.svhn_loader as svhn
    import utils.lsun_loader as lsun_loader

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--dataset', '-d', type=str, default='cifar10', help='Dataset name.')
parser.add_argument('--model', '-model', type=str, default='wrn', help='Architecture name.')
# Loading details
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--load', '-l', type=str, default='./snapshots', help='Checkpoint path to resume / test.')
parser.add_argument('--gpu', type=int, action='append', default=[], help='Which gpus to use.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
args = parser.parse_args()

experiment = Experiment(api_key="T1ICBKLfUXrSnfizBvUW2K0GA", project_name="msc-thesis-ai", workspace="awgansekoele",
                        log_graph=False, parse_args=False)
experiment.log_parameters(vars(args))

# torch.manual_seed(1)
# np.random.seed(1)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if 'cifar10' == args.dataset:
    test_data = dset.CIFAR10('/raid/data/arwin/data', train=False, transform=test_transform)
    num_classes = 10
else:
    test_data = dset.CIFAR100('/raid/data/arwin/data', train=False, transform=test_transform)
    num_classes = 100

test_loader = DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                         num_workers=args.prefetch, pin_memory=True)

# Create model
if 'allconv' in args.method_name:
    net = AllConvNet(num_classes)
else:
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

experiment.set_model_graph(str(net), overwrite=True)

start_epoch = 0

# Restore model
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        subdir = 'oe_scratch'

        model_name = os.path.join(os.path.join(args.load, subdir),
                                  args.dataset + '_' + args.model + '_oe_scratch_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume"
    else:
        experiment.log_model("model", model_name)

net.eval()

if len(args.gpu) > 1:
    net = torch.nn.DataParallel(net, device_ids=args.gpu)

if len(args.gpu) > 0:
    device = torch.device('cuda:{}'.format(args.gpu[0]))
    net.to(device)
    torch.cuda.manual_seed(1)
else:
    device = torch.device('cpu')

cudnn.benchmark = True  # fire on all cylinders

# /////////////// Detection Prelims ///////////////

ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.to(device)

            output = net(data)
            smax = to_np(F.softmax(output, dim=1))

            if args.use_xent:
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                _score.append(-np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if args.use_xent:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

num_right = len(right_score)
num_wrong = len(wrong_score)
print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))

# /////////////// End Detection Prelims ///////////////

print('\nUsing CIFAR-10 as typical data') if num_classes == 10 else print('\nUsing CIFAR-100 as typical data')

# /////////////// Error Detection ///////////////

print('\n\nError Detection')
show_performance(wrong_score, right_score, method_name=args.dataset + '_' + args.model + '_oe_scratch')

# /////////////// OOD Detection ///////////////
auroc_list, aupr_list, fpr_list = [], [], []


def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):
    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader)
        measures = get_measures(out_score, in_score)
        aurocs.append(measures[0]);
        auprs.append(measures[1]);
        fprs.append(measures[2])

    auroc = np.mean(aurocs);
    aupr = np.mean(auprs);
    fpr = np.mean(fprs)
    auroc_list.append(auroc);
    aupr_list.append(aupr);
    fpr_list.append(fpr)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, args.dataset + '_' + args.model + '_oe_scratch')
    else:
        print_measures(auroc, aupr, fpr, args.dataset + '_' + args.model + '_oe_scratch')


def get_adv_loader(adv_images, adv_labels):
    nr_images = adv_images.size(0)
    correct = 0

    adv_data = TensorDataset(adv_images.float() / 255, adv_labels)
    adv_loader = DataLoader(adv_data, batch_size=args.test_bs, shuffle=False)

    wrong_data, wrong_target = None, None

    for data, target in adv_loader:
        data, target = data.to(device), target.to(device)
        output = net(data)

        iseq = output.argmax(dim=1) == target
        correct += iseq.sum()
        wrong_data = data[~iseq] if wrong_data is None else torch.cat((wrong_data, data[~iseq]))
        wrong_target = target[~iseq] if wrong_target is None else torch.cat((wrong_target, target[~iseq]))

    wrong_data = TensorDataset(wrong_data, wrong_target)
    return DataLoader(wrong_data, batch_size=args.test_bs, shuffle=False), correct.float() / nr_images


# /////////////// FGSM Noise ///////////////

adv_images, adv_labels = torch.load(os.path.join('/raid/data/arwin/outlier-exposure/CIFAR/snapshots/adv_attacks/',
                                                 args.dataset + '_' + args.model + '_baseline_fgsm.pt'))
adv_loader, acc = get_adv_loader(adv_images, adv_labels)

print('\n\nFGSM Perturbed Image Detection')
print('Accuracy: ' + str(acc.item() * 100))
get_and_print_results(adv_loader)

# /////////////// BIM Noise ///////////////

adv_images, adv_labels = torch.load(os.path.join('/raid/data/arwin/outlier-exposure/CIFAR/snapshots/adv_attacks/',
                                                 args.dataset + '_' + args.model + '_baseline_bim.pt'))
adv_loader, acc = get_adv_loader(adv_images, adv_labels)

print('\n\nBIM Perturbed Image Detection')
print('Accuracy: ' + str(acc.item() * 100))
get_and_print_results(adv_loader)

# /////////////// CW Noise ///////////////

adv_images, adv_labels = torch.load(os.path.join('/raid/data/arwin/outlier-exposure/CIFAR/snapshots/adv_attacks/',
                                                 args.dataset + '_' + args.model + '_baseline_cw.pt'))
adv_loader, acc = get_adv_loader(adv_images, adv_labels)

print('\n\nCW Perturbed Image Detection')
print('Accuracy: ' + str(acc.item() * 100))
get_and_print_results(adv_loader)

# /////////////// DeepFool Noise ///////////////

adv_images, adv_labels = torch.load(os.path.join('/raid/data/arwin/outlier-exposure/CIFAR/snapshots/adv_attacks/',
                                                 args.dataset + '_' + args.model + '_baseline_deepfool.pt'))
adv_loader, acc = get_adv_loader(adv_images, adv_labels)

print('\n\nDeepFool Perturbed Image Detection')
print('Accuracy: ' + str(acc.item() * 100))
get_and_print_results(adv_loader)

# /////////////// PGD Noise ///////////////

adv_images, adv_labels = torch.load(os.path.join('/raid/data/arwin/outlier-exposure/CIFAR/snapshots/adv_attacks/',
                                                 args.dataset + '_' + args.model + '_baseline_pgd.pt'))
adv_loader, acc = get_adv_loader(adv_images, adv_labels)

print('\n\nPGD Perturbed Image Detection')
print('Accuracy: ' + str(acc.item() * 100))
get_and_print_results(adv_loader)

# /////////////// Mean Results ///////////////

print('\n\nMean Test Results')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.dataset + '_' + args.model + '_oe_scratch')
