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
import torchattacks
from models.allconv import AllConvNet
from models.wrn import WideResNet
from models.distnet import DistanceNet
from skimage.filters import gaussian as gblur
from PIL import Image as PILImage

from torch.utils.data import TensorDataset

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--method_name', '-m', type=str, default='cifar10_allconv_baseline', help='Method name.')
# Loading details
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--load', '-l', type=str, default='./snapshots', help='Checkpoint path to resume / test.')
parser.add_argument('--save', '-s', type=str, default='./snapshots/adv_attacks', help='Folder to save attacks.')
parser.add_argument('--gpu', type=int, action='append', default=[], help='Which gpus to use.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
args = parser.parse_args()

experiment = Experiment(api_key="T1ICBKLfUXrSnfizBvUW2K0GA", project_name="msc-thesis-ai", workspace="awgansekoele",
                        log_graph=False, parse_args=False)
experiment.log_parameters(vars(args))

torch.manual_seed(1)
np.random.seed(1)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if 'cifar10_' in args.method_name:
    test_data = dset.CIFAR10('/raid/data/arwin/data', train=False, transform=test_transform)
    num_classes = 10
else:
    test_data = dset.CIFAR100('/raid/data/arwin/data', train=False, transform=test_transform)
    num_classes = 100

test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)

# Create model
if 'allconv' in args.method_name:
    net = AllConvNet(num_classes)
else:
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

experiment.set_model_graph(str(net), overwrite=True)

start_epoch = 0

if not os.path.exists(args.save):
    os.makedirs(args.save)

# Restore model
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        if 'baseline' in args.method_name:
            subdir = 'baseline'
        elif 'oe_tune' in args.method_name:
            subdir = 'oe_tune'
        else:
            subdir = 'oe_scratch'

        model_name = os.path.join(os.path.join(args.load, subdir), args.method_name + '_epoch_' + str(i) + '.pt')
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

if not os.path.exists(args.save):
    os.makedirs(args.save)

### Create Tensor Dataset with only correct images ###

cpu = torch.device('cpu')
cor_data, cor_target = None, None

for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    output = net(data)
    iseq = output.argmax(dim=1) == target
    cor_data = data[iseq].to(cpu) if cor_data is None else torch.cat((cor_data, data[iseq].to(cpu)))
    cor_target = target[iseq].to(cpu) if cor_target is None else torch.cat((cor_target, target[iseq].to(cpu)))

test_data = TensorDataset(cor_data, cor_target)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False)

#### FGSM Attack ####

print('\nPerforming FGSM attack.')
fgsm_attack = torchattacks.FGSM(net)
fgsm_attack.set_mode('int')
fgsm_attack.save(data_loader=test_loader, file_name=os.path.join(args.save, args.method_name +
                                                                 '_fgsm.pt'), accuracy=True)

### BIM Attack ###

print('\nPerforming BIM attack.')
bim_attack = torchattacks.BIM(net)
bim_attack.set_mode('int')
bim_attack.save(data_loader=test_loader, file_name=os.path.join(args.save, args.method_name +
                                                                '_bim.pt'), accuracy=True)

### CW Attack ###

print('\nPerforming CW attack with c=1.')
cw_attack = torchattacks.CW(net, c=1)
cw_attack.set_mode('int')
cw_attack.save(data_loader=test_loader, file_name=os.path.join(args.save, args.method_name +
                                                               '_cw.pt'), accuracy=True)

### DeepFool Attack ###

print('\nPerforming DeepFool attack.')
deepfool_attack = torchattacks.DeepFool(net)
deepfool_attack.set_mode('int')
deepfool_attack.save(data_loader=test_loader, file_name=os.path.join(args.save, args.method_name +
                                                                     '_deepfool.pt'), accuracy=True)

### PGD Attack ###

print('\nPerforming PGD Attack')
pgd_attack = torchattacks.PGD(net)
pgd_attack.set_mode('int')
pgd_attack.save(data_loader=test_loader, file_name=os.path.join(args.save, args.method_name +
                                                                '_pgd.pt'), accuracy=True)
