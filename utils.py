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
    i = max([int(model[-1]) for model in models]) + 1
    return f'{t}_{i}'

def logging_setup(args, name):
    logger = logging.getLogger(name)
    logging.basicConfig(filename = os.path.join(args.savepath, 'train.log'), level=logging.INFO)
    logger.info(f'Model ID: {args.modelID}')
    logger.info(f'Training file: {args.data}, normalisation: {args.normalize}')
    logger.info(f'Model used: {args.model}')
    logger.info(f'Number of graph nodes: {args.n_e}')
    logger.info(f'GNN: channel size {args.channel_size}, layer 1 {args.hid1}, layer 2 {args.hid2}, highway window {args.highway_window}')
    logger.info(f'CNN: kernel sizes {args.k_size}, window size {args.window}')
    logger.info(f'Optimiser: {args.optim}, learning rate {args.lr}, gradient clipping {args.clip}')