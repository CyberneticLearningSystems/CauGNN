import os
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from typing import Dict
from DataUtility import DataUtility

class AirlineData():
    def __init__(self, args, train: float, test: float, rawdat: pd.DataFrame):
        self.cuda: bool = args.cuda
        self.horizon: int = args.horizon
        self.window: int = args.window
        self.normalize: int = args.normalize
        self.ncols: int = rawdat.shape[1]
        self.dat: Dict[np.ndarray[float]]
        self.Data = Dict[DataUtility]
        self.train: Dict[list[torch.Tensor]]
        self.test: Dict[list[torch.Tensor]]
        self.valid: Dict[list[torch.Tensor]]
        self._airline_batching(args.data, train, test)
    

    def _airline_batching(self, df: pd.DataFrame, train: float, test: float):
        self.airlines: list[str] = df['UNIQUE_CARRIER_NAME'].unique()
        self.nairlines: int = len(self.airlines)
        #? Do we only want to take airlines which have identical length of timeseries? Yes --> ndarray, No --> dict 
        for airline in enumerate(self.airlines):
            rawdat = df[df['UNIQUE_CARRIER_NAME'] == airline]
            rawdat.drop(columns=['UNIQUE_CARRIER_NAME'], inplace=True)
            self.Data[airline] = DataUtility(rawdat, self.cuda, self.horizon, self.window, self.normalize)
