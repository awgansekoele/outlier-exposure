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
from torchvision.models import resnet18
import torch.nn.functional as F
from models.distnet import PrioriNet
from skimage.filters import gaussian as gblur
from PIL import Image as PILImage

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
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
# Loading details
parser.add_argument('--z-dim', default=10, type=int, help='latent dimension')
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

test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)

net = resnet18(pretrained=True)
net.fc = nn.Linear(net.fc.in_features, args.z_dim)
net.fc.reset_parameters()
net = PrioriNet(backbone=net, z_dim=args.z_dim, n_classes=num_classes)
experiment.set_model_graph(str(net), overwrite=True)

start_epoch = 0

# Restore model
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        subdir = 'baseline2'

        model_name = os.path.join(os.path.join(args.load, subdir), args.dataset + '_resnet18_baseline_epoch_' + str(i)
                                  + '_1.pt')
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

            _score.append(-to_np(output.data.max(1).values))
            if in_dist:
                preds = np.argmax(to_np(output), axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                _right_score.append(-to_np(output.data.max(1).values)[right_indices])
                _wrong_score.append(-to_np(output.data.max(1).values)[wrong_indices])

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
show_performance(wrong_score, right_score, method_name=args.dataset + '_resnet18_baseline')

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
        print_measures_with_std(aurocs, auprs, fprs, args.dataset + '_resnet18_baseline')
    else:
        print_measures(auroc, aupr, fpr, args.dataset + '_resnet18_baseline')


# /////////////// Gaussian Noise ///////////////

dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
ood_data = torch.from_numpy(np.float32(np.clip(
    np.random.normal(size=(ood_num_examples * args.num_to_avg, 3, 32, 32), scale=0.5), -1, 1)))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nGaussian Noise (sigma = 0.5) Detection')
get_and_print_results(ood_loader)

# /////////////// Rademacher Noise ///////////////

dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
ood_data = torch.from_numpy(np.random.binomial(
    n=1, p=0.5, size=(ood_num_examples * args.num_to_avg, 3, 32, 32)).astype(np.float32)) * 2 - 1
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nRademacher Noise Detection')
get_and_print_results(ood_loader)

# /////////////// Blob ///////////////

ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(ood_num_examples * args.num_to_avg, 32, 32, 3)))
for i in range(ood_num_examples * args.num_to_avg):
    ood_data[i] = gblur(ood_data[i], sigma=1.5, multichannel=False)
    ood_data[i][ood_data[i] < 0.75] = 0.0

dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
ood_data = torch.from_numpy(ood_data.transpose((0, 3, 1, 2))) * 2 - 1
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nBlob Detection')
get_and_print_results(ood_loader)

# /////////////// Textures ///////////////

ood_data = dset.ImageFolder(root='/raid/data/arwin/data/textures/dtd/images',
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nTexture Detection')
get_and_print_results(ood_loader)

# /////////////// SVHN ///////////////

ood_data = svhn.SVHN(root='/raid/data/arwin/data/svhn', split="test",
                     transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]), download=False)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nSVHN Detection')
get_and_print_results(ood_loader)

# /////////////// Places365 ///////////////

# ood_data = dset.ImageFolder(root="/share/data/vision-greg2/places365/test_subset", # /raid/data/arwin/data/svhn
#                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
#                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
# ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
#                                         num_workers=args.prefetch, pin_memory=True)

# print('\n\nPlaces365 Detection')
# get_and_print_results(ood_loader)

# /////////////// LSUN ///////////////

ood_data = lsun_loader.LSUN("/raid/data/arwin/data/lsun", classes='test',
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nLSUN Detection')
get_and_print_results(ood_loader)

# /////////////// CIFAR Data ///////////////

if 'cifar10'== args.dataset:
    ood_data = dset.CIFAR100('/raid/data/arwin/data', train=False, transform=test_transform)
else:
    ood_data = dset.CIFAR10('/raid/data/arwin/data', train=False, transform=test_transform)

ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nCIFAR-100 Detection') if 'cifar100' == args.dataset else print('\n\nCIFAR-10 Detection')
get_and_print_results(ood_loader)

# /////////////// Mean Results ///////////////

print('\n\nMean Test Results')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.dataset + '_resnet18_baseline')