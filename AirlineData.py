import os
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from typing import Dict
from DataUtility import DataUtility

class AirlineData():
    def __init__(self, args, train: float, test: float, rawdat: pd.DataFrame):
        self.args = args
        self.Data = Dict[DataUtility]
        self._airline_batching(rawdat, train, test)
    

    def _airline_batching(self, df: pd.DataFrame, train: float, test: float):
        self.airlines: list[str] = df['AIRLINE_ID'].unique()
        self.nairlines: int = len(self.airlines)
        for airline in enumerate(self.airlines):
            airlinedat = df[df['AIRLINE_ID'] == airline]
            airlinedat.drop(columns=['AIRLINE_ID'], inplace=True)
            self.Data[airline] = DataUtility(self.args, train, test, airlinedat)
