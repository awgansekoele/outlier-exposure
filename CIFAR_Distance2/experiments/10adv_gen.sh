#!/bin/bash

python adv_gen.py --method_name cifar10_wrn_oe_scratch --z-dim 1024 > snapshots/cifar10_wrn_oe_scratch_adv_gen.txt
# allconv_oe_scratch, wrn_baseline, wrn_oe_tune