import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable


def normal_std(x) -> float:
    """
    Standard Deviation with Bessels Correction, i.e. calculating the standard deviation from a sample of a population.
    """
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))


def load_form41(path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(path, delimiter=',')
        _ = data['YEAR']
    except KeyError:
        data = pd.read_csv(path, delimiter=';')
        _ = data['YEAR']
    return data


def form41_dataloader(path: str, batching: bool = False) -> pd.DataFrame:
    '''
    Loads Form41 data for CauGNN training, differentiating between datasets with multiple airlines and single-airline datasets. For single-airline datasets,
    the UNIQUE_CARRIER_NAME column is dropped, for mutli-airline datasets it is retained for later batching. 
    '''
    data = load_form41(path)
    data.dropna(inplace=True, axis=0)
    if batching: 
         data.drop(columns=['UNIQUE_CARRIER_NAME', 'YEAR', 'QUARTER'], inplace=True)
    else:
        data.drop(columns=['AIRLINE_ID', 'YEAR', 'QUARTER', 'UNIQUE_CARRIER_NAME'], inplace=True)
    return data


def dataloader(path: str):
    try:
        data = pd.read_csv(path, delimiter=',')
        datetimes = list(pd.to_datetime(data.pop('date')))
    except KeyError:
        data = pd.read_csv(path, delimiter=';')
        datetimes = list(pd.to_datetime(data.pop('date'), format='%d.%m.%Y %H:%M'))
    data = np.array(data, dtype=float)
    return data
