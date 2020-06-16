# -*- coding: utf-8 -*-
from comet_ml import Experiment

import numpy as np
import os
import pickle
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
from torchvision.models import resnet18
import torch.nn.functional as F
from tqdm import tqdm
from models.distnet import HypersphericalNet

# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.validation_dataset import validation_split

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--z-dim', default=1024, type=int, help='latent dimension')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/baseline3', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--seed', type=int, default=1, help='The seed to use.')
parser.add_argument('--gpu', type=int, action='append', default=[], help='Which gpus to use.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

experiment = Experiment(api_key="T1ICBKLfUXrSnfizBvUW2K0GA", project_name="msc-thesis-ai", workspace="awgansekoele",
                        log_graph=False, parse_args=False)
experiment.log_parameters(vars(args))

torch.manual_seed(1)
np.random.seed(1)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if args.dataset == 'cifar10':
    train_data = dset.CIFAR10('/raid/data/arwin/data', train=True, transform=train_transform)
    test_data = dset.CIFAR10('/raid/data/arwin/data', train=False, transform=test_transform)
    num_classes = 10
else:
    train_data = dset.CIFAR100('/raid/data/arwin/data', train=True, transform=train_transform)
    test_data = dset.CIFAR100('/raid/data/arwin/data', train=False, transform=test_transform)
    num_classes = 100

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

net = resnet18(pretrained=True)
net.fc = nn.Linear(net.fc.in_features, args.z_dim)
net.fc.reset_parameters()
net = HypersphericalNet(backbone=net, z_dim=args.z_dim, n_classes=num_classes)
experiment.set_model_graph(str(net), overwrite=True)

start_epoch = 0

# Restore model if desired
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        model_name = os.path.join(args.load, args.dataset + '_' + args.model +
                                  '_baseline_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume"

if len(args.gpu) > 1:
    net = torch.nn.DataParallel(net, device_ids=args.gpu)

if len(args.gpu) > 0:
    device = torch.device('cuda:{}'.format(args.gpu[0]))
    net.to(device)
    torch.cuda.manual_seed(1)
else:
    device = torch.device('cpu')

cudnn.benchmark = True  # fire on all cylinders

optimizer = torch.optim.SGD(
    net.parameters(), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))


# /////////////// Training ///////////////

def train():
    with experiment.train():
        net.train()  # enter train mode
        loss_avg = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            # forward
            x = net(data)

            # backward
            scheduler.step()
            optimizer.zero_grad()
            loss = torch.gather(1 - x, 1, target.view(-1, 1)).mean()
            loss.backward()
            optimizer.step()

            # exponential moving average
            loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        state['train_loss'] = loss_avg
        experiment.log_metric('loss', state['train_loss'])


# test function
def test():
    with experiment.test():
        net.eval()
        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                # forward
                output = net(data)
                loss = torch.gather(1 - output, 1, target.view(-1, 1)).mean()

                # accuracy
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                loss_avg += float(loss.data)

        state['test_loss'] = loss_avg / len(test_loader)
        state['test_accuracy'] = correct / len(test_loader.dataset)
        experiment.log_metric('loss', state['test_loss'])
        experiment.log_metric('accuracy', state['test_accuracy'])


if args.test:
    test()
    print(state)
    exit()

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

with open(os.path.join(args.save, args.dataset + '_' + 'resnet18' +
                                  '_baseline_training_results_' + str(args.seed) + '.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')

# Main loop
for epoch in range(start_epoch, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train()
    test()

    # Save model
    torch.save(net.state_dict(),
               os.path.join(args.save, args.dataset + '_' + 'resnet18' +
                            '_baseline_epoch_' + str(epoch) + '_' + str(args.seed) + '.pt'))
    # Let us not waste space and delete the previous model
    prev_path = os.path.join(args.save, args.dataset + '_' + 'resnet18' +
                             '_baseline_epoch_' + str(epoch - 1) + '_' + str(args.seed) + '.pt')
    if os.path.exists(prev_path): os.remove(prev_path)

    # Show results

    with open(os.path.join(args.save, args.dataset + '_' + 'resnet18' +
                                      '_baseline_training_results_' + str(args.seed) + '.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'],
        ))

    # # print state with rounded decimals
    # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100 - 100. * state['test_accuracy'])
    )

    experiment.log_model("model", os.path.join(args.save, args.dataset + '_' + 'resnet18' +
                                               '_baseline_epoch_' + str(epoch) + '_' + str(args.seed) + '.pt'))
    experiment.log_asset(os.path.join(args.save, args.dataset + '_' + 'resnet18' +
                                      '_baseline_training_results_' + str(args.seed) + '.csv'))
