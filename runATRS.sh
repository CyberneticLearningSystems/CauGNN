#!/usr/bin/env bash

python main.py --model TENet --window 8 --horizon 4 --batch_size 32 --channel_size 16 --hid1 30 --hid2 20 --lr 0.0011960721309568513 --epochs 100 --data ./Data/form41_aggregated_quarterly_reduced_LargeUSC.csv --A ./TENet_master/TE/form41_aggregated_quarterly_USC_reduced_TE.txt --form41 True --airline_batching True --sharedTE True --seed 123 --cuda '' --print True --tune ''