import os
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from typing import Dict
from DataUtility import DataUtility
from TENet_master.util import Teoriginal

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
            airlinedat = airlinedat.drop(columns=['AIRLINE_ID'])
            self.Airlines[airline] = DataUtility(self.args, train, airlinedat)
            self.airlines.append(airline)
            self.nairlines += 1

    def airline_matrix(self, airline: str):
        savepath = os.path.join(os.path.dirname(self.args.data), 'causality_matrix', f'{os.path.basename(self.args.data).split(".")[0]}_{airline}_TE.txt')
        if self.args.sharedTE:
            A = np.loadtxt(self.args.A)
            A = np.array(A, dtype=np.float32)
            return A
        elif os.path.exists(savepath):
            print(f'Transfer Entropy matrix for {airline} already exists at {savepath}')
            return np.loadtxt(savepath)
        else: 
            print(f'Calculating Transfer Entropy matrix for airline {airline}')
            A = Teoriginal._te_calculation(self.Airlines[airline].dat)
            Teoriginal._save_matrix(A, savepath)
            print(f'Transfer Entropy matrix for airline {airline} saved at {savepath}')
            return A
