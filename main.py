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
import json
import pickle
import multiprocessing

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
from ray.tune.error import TuneError
from pathlib import Path


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
    parser.add_argument('--tune_trials', type=int, default=10, help='Number of trials for hyperparameter tuning')
    args = parser.parse_args()

    if args.tune:
        config = {
            "hid1": tune.choice([40,30,20,10,5]),
            "hid2": tune.choice([25,20,15,10,5]),
            "lr": tune.loguniform(1e-4, 1e-1),
            "channel_size": tune.choice([2**i for i in range(9)]),
            "batch_size": tune.choice([4,8,12,16,20,24,32]),
        }

        args.A = os.path.abspath(args.A) #* ray tune requires absolute path as it changes the working directory to a new directory
        args.data = os.path.abspath(args.data)

        if args.airline_batching:

            caugnn = CauGNN(args)
            number_of_airlines = len(caugnn.Data.airlines)
            number_of_trials = args.tune_trials
            gpus_per_trial = torch.cuda.device_count()/number_of_trials
            cpus_per_trial = math.floor((multiprocessing.cpu_count())-2/number_of_trials)
            

            # RayTune ASHA Scheduler runs maximum of training iterations equal to the number of airlines times the number of epochs per airline
            # The minimum number of iterations (grace_periods) is set to 5 times the number of epochs per airline --> thus the ASHA scheudler will run through at least 5 airlines
            scheduler = ASHAScheduler(
                metric="Training Loss",
                mode="min",
                max_t=number_of_airlines*args.epochs,
                grace_period=5*args.epochs,
                reduction_factor=2,
            )

            result = tune.run(
                caugnn.run_airline_training,
                resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
                config=config,
                num_samples=number_of_trials, # Number of trials (different sets of hyperparameters to run)
                scheduler=scheduler,
                checkpoint_at_end=False, #Checkpoint is already saved after each epoch, see caugnn.run_epoch()
                local_dir=os.path.abspath('./models/ray_results'),
                fail_fast=False, # Continue even if a trial fails
                raise_on_failed_trial=False # Writes to the variable result even if a trial fails
            )
                    
            best_trial = result.get_best_trial("Training Loss", "min", "last")
            print(f"Best trial config: {best_trial.config}")
            print(f"Best trial final training loss: {best_trial.last_result['Training Loss']}")
            print(f"Best trial final test MAE: {best_trial.last_result['Test MAE']}")

            print('\n \n \nTRAINING COMPLETE - SAVE BEST MODEL')

            # Load the best checkpoint
            best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="Test MAE", mode="min")
            with best_checkpoint.as_directory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "rb") as fp:
                    best_checkpoint_data = pickle.load(fp)

            # Save the best trial's model state dict & config 
            best_model_state_dict = best_checkpoint_data["net_state_dict"]     
            save_dir = os.path.abspath('./models/best_model')
            os.makedirs(save_dir, exist_ok=True)

            # Save the best trained model
            model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(best_model_state_dict, model_path)

            # Save the best trial's config
            config_path = os.path.join(save_dir, 'best_config.json')
            with open(config_path, 'w') as f:
                json.dump(best_trial.config, f)
            print('SCRIPT COMPLETED \n \n \n')           

        else:
            print(f'\n \n \nHyperparameter Tuning not supported for individual airline training.\n \n \n')

    else:
        caugnn = CauGNN(args)
        config = None        
        if args.airline_batching:
            caugnn.run_airline_training(config)
        else:
            caugnn.run_training(caugnn.Data)


