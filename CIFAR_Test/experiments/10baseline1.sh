#!/bin/bash

for SEED in 1 2 3
do
  python baseline1.py cifar10 --z-dim 1024 --seed $SEED #-c
done

