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
from vis import *
from CauGNN import CauGNN


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multivariate Time series forecasting')
    parser.add_argument('--data', type=str, default=None, help='location of the data file')
    parser.add_argument('--n_e', type=int, default=None, help='The number of graph nodes')
    parser.add_argument('--model', type=str, default='TENet', help='Model type to use')
    parser.add_argument('--k_size', type=list, default=[3,5,7], help='number of CNN kernel sizes', nargs='*')
    parser.add_argument('--window', type=int, default=32, help='window size')
    parser.add_argument('--decoder', type=str, default= 'GNN', help = 'type of decoder layer')
    parser.add_argument('--horizon', type=int, default= 5, help = 'forecasting horizon')
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
    parser.add_argument('--print', type=bool, default=False, help='prints the evaluation metric while training')
    parser.add_argument('--airline_batching', type=bool, default=False, help='Batch data by airline')
    args = parser.parse_args()


    caugnn = CauGNN(args)
    if args.airline_batching:
        caugnn.run_airline_training()
    else:
        caugnn.run_training()