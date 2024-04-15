#!/usr/bin/env bash

python train.py --model TENet --epochs 10 --window 32 --horizon 5 --highway_window 1 --channel_size 12 --hid1 40 --hid2 10 --data Data/exchange_rate.csv --n_e 8 --A TENet_master/TE/exte.txt --save Model/model.pt 








