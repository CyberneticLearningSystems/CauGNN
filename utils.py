import pandas as pd
import numpy as np
import time
import os
import logging

def set_modelID():
    path = 'models/'
    t = str(pd.to_datetime('today').date())
    models = os.listdir(path)
    models = [model for model in models if model.startswith(t)]
    if models:
        i = max([int(model[-1]) for model in models]) + 1
    else:
        i = 1
    return f'{t}_{i}'

def model_logging(args, path):
    # save model parameters into .txt
    with open(os.path.join(path, 'modelinfo.txt'), 'w') as f:
        f.write(f'Model ID: {args.modelID}\n')
        f.write(f'Training file: {args.data}, normalisation: {args.normalize}\n')
        f.write(f'Model used: {args.model}\n')
        f.write(f'Training Data Size: {args.train}, Number of Epoch {args.epochs}\n')
        f.write(f'Number of graph nodes: {args.n_e}\n')
        f.write(f'GNN: channel size {args.channel_size}, layer 1 {args.hid1}, layer 2 {args.hid2}, highway window {args.highway_window}\n')
        f.write(f'CNN: kernel sizes {args.k_size}, window size {args.window}, horizon {args.horizon}, batch size {args.batch_size}, dropout {args.dropout}\n')
        f.write(f'Optimiser: {args.optim}, learning rate {args.lr}, gradient clipping {args.clip}')
        f.close()