#!/bin/bash

python adv_test_black.py --dataset cifar10 --num_to_avg 10 > snapshots/cifar10_wrn_oe_scratch_adv_test_black.txt
# allconv_oe_scratch, wrn_baseline, wrn_oe_tune
