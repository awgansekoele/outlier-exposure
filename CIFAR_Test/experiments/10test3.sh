#!/bin/bash

python test3.py cifar10 --num_to_avg 10 --z-dim 1024 > snapshots/cifar10_resnet18_baseline3_test.txt
# allconv_oe_scratch, wrn_baseline, wrn_oe_tune

