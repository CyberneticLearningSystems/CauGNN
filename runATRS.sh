#!/usr/bin/env bash

python main.py --model TENet --window 8 --horizon 4 --batch_size 16 --channel_size 12 --hid1 40 --hid2 10 --epochs 10 --data ./Data/form41_aggregated_quarterly_reduced_LargeUSC.csv --A ./TENet_master/TE/form41_aggregated_quarterly_USC_reduced_TE.txt --form41 True --airline_batching True --sharedTE True --seed 123 --cuda '' --print True --tune ''