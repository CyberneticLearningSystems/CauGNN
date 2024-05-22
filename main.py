import argparse
import math
import time
import Optim
import torch
import torch.nn as nn
import numpy as np
import importlib
import sys
import os
import pickle
import logging
import utils

import data_utils
from DataUtility import DataUtility
from AirlineData import AirlineData
# from ml_eval import *
from TENet_master.models import *
from TENet_master.util import Teoriginal
from eval import evaluate
np.seterr(divide='ignore',invalid='ignore')
from TENet_master.models import TENet
from CauGNN import CauGNN
from ray import tune
from ray.tune.schedulers import ASHAScheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multivariate Time series forecasting')
    parser.add_argument('--data', type=str, default=None, help='location of the data file')
    parser.add_argument('--n_e', type=int, default=None, help='The number of graph nodes')
    parser.add_argument('--model', type=str, default='TENet', help='Model type to use')
    parser.add_argument('--k_size', type=list, default=[3,5,7], help='number of CNN kernel sizes', nargs='*')
    parser.add_argument('--window', type=int, default=8, help='window size')
    parser.add_argument('--decoder', type=str, default= 'GNN', help = 'type of decoder layer')
    parser.add_argument('--horizon', type=int, default= 4, help = 'forecasting horizon')
    parser.add_argument('--A', type=str, default=None, help='Adjenency matrix, calculated from Transfer Entropy')
    parser.add_argument('--highway_window', type=int, default=1, help='The window size of the highway component')
    parser.add_argument('--channel_size', type=int, default=12, help='the channel size of the CNN layers')
    parser.add_argument('--hid1', type=int, default=40, help='the hidden size of the GNN layers')
    parser.add_argument('--hid2', type=int, default=10, help='the hidden size of the GNN layers')
    parser.add_argument('--clip', type=float, default=10, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=54321, help='random seed')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
    parser.add_argument('--modelID', type=str,  default=None, help='model ID')
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--L1Loss', type=bool, default=True)
    parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--output_fun', type=str, default=None)
    parser.add_argument('--num_adj', type=int, default=1)
    parser.add_argument('--attention_mode', type=str, default='naive')
    parser.add_argument('--skip_mode', type=str, default='concat')
    parser.add_argument('--form41', type=bool, default=False)
    parser.add_argument('--print', type=bool, default=False, help='prints the evaluation metrics after training')
    parser.add_argument('--printc', type=bool, default=False, help='prints the evaluation metrics while training')
    parser.add_argument('--airline_batching', type=bool, default=False, help='Batch data by airline')
    parser.add_argument('--sharedTE', type=bool, default=False, help='Use shared TE matrix, i.e. same TE matrix for all airlines')
    parser.add_argument('--tune', type=bool, default=False, help='Hyperparameter tuning with ray tune')
    args = parser.parse_args()

    if args.tune:
        config = {
            "hid1": tune.choice([40,30,20,10,5]),
            "hid2": tune.choice([25,20,15,10,5]),
            "lr": tune.loguniform(1e-4, 1e-1),
            "channel_size": tune.choice([2**i for i in range(9)]),
            "batch_size": tune.choice([4, 8, 16, 32, 64]),
        }

        args.A = os.path.abspath(args.A) #* ray tune requires absolute path as it changes the working directory to a new directory
        args.data = os.path.abspath(args.data)

        if args.airline_batching:

            caugnn = CauGNN(args)
            gpus_per_trial = torch.cuda.device_count()

            scheduler = ASHAScheduler(
                metric="Training Loss",
                mode="min",
                max_t=args.epochs,
                grace_period=1,
                reduction_factor=2,
            )

            result = tune.run(
                caugnn.run_airline_training,
                resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
                config=config,
                num_samples=10,
                scheduler=scheduler,
                checkpoint_at_end=False,
                local_dir=os.path.abspath('./models/ray_results')
            )
            
            best_trial = result.get_best_trial("Training Loss", "min", "last")
            print(f"Best trial config: {best_trial.config}")
            print(f"Best trial final training loss: {best_trial.last_result['Training Loss']}")
            print(f"Best trial final test RMSE: {best_trial.last_result['Test RMSE']}")
                        
        else:
            print(f'\n \n \nHyperparameter Tuning not supported for individual airline training.\n \n \n')
    
    else:
        caugnn = CauGNN(args)
        config = None        
        if args.airline_batching:
            caugnn.run_airline_training(config)
        else:
            caugnn.run_training(caugnn.Data)



