import os
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from typing import Dict
from DataUtility import DataUtility

class AirlineData():
    def __init__(self, args, train: float, rawdat: pd.DataFrame):
        self.args = args
        self.Airlines: Dict[str, DataUtility] = {}
        self._airline_batching(rawdat, train)
    

    def _airline_batching(self, df: pd.DataFrame, train: float):
        airlines: list[str] = df['AIRLINE_ID'].unique()
        self.airlines = []
        self.nairlines: int = 0
        for airline in airlines:
            airlinedat = df[df['AIRLINE_ID'] == airline]
            if len(airlinedat) < 80:
                continue
            airlinedat = airlinedat.drop(columns=['AIRLINE_ID'], inplace=True)
            self.Airlines[airline] = DataUtility(self.args, train, airlinedat)
            self.airlines.append(airline)
            self.nairlines += 1
