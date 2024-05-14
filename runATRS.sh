#!/usr/bin/env bash

python main.py --model TENet --window 8 --horizon 4 --batch_size 16 --channel_size 12 --hid1 40 --hid2 10 --epoch 10 --data ./Data/form41_aggregated_monthly_reduced_dataset_LargeUSC.csv --A ./TENet_master/TE/form41_aggregated_quarterly_USC_reduced_TE.txt --form41 True --airline_batching True --seed 123 --cuda True