import os
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from typing import Dict

class AirlineData():
    def __init__(self, args, train: float, test: float):
        self.cuda: bool = args.cuda
        self.horizon: int = args.horizon
        self.window: int = args.window
        self.normalize: int = args.normalize
        self.data: np.ndarray = np.ndarray((0, 0))
        self.datetimes: list[pd.Timestamp] = []
        self.rows: int = 0
        self.cols: int = 0

        self.dat: Dict[np.ndarray[float]]
        self._airline_batching(args.data, train, test)

        self.train: Dict[list[torch.Tensor]]
        self.test: Dict[list[torch.Tensor]]
        self.valid: Dict[list[torch.Tensor]]

        # TODO: implement batch loader

        #! Code copied from data_utils.py
        self.scale: np.ndarray = np.ones(self.cols)
        self._normalise()
        self._split(int(train * self.rows), int((train+test) * self.rows))


    def _form41_dataloader(self, path) -> pd.DataFrame:
        try:
            data = pd.read_csv(path, delimiter=',')
            self.year = data.pop('YEAR')
        except KeyError:
            data = pd.read_csv(path, delimiter=';')
            self.year = data.pop('YEAR')
        return data
    

    def _airline_batching(self, datapath: str, train: float, test: float):
        data = self._form41_dataloader(datapath)
        self.airlines = data['UNIQUE_CARRIER_NAME'].unique()
        self.nairlines = len(self.airlines)
        #? Do we only want to take airlines which have identical length of timeseries? Yes --> ndarray, No --> dict 
        # self.data: np.ndarray = np.ndarray((len(airlines), self.rows, self.cols))
        self.data: dict = {}
        for airline in enumerate(self.airlines):
            rawdat = data[data['UNIQUE_CARRIER_NAME'] == airline]
            rawdat.drop(columns=['UNIQUE_CARRIER_NAME'], inplace=True)
            rawdat.dropna(inplace=True, axis=0)
            ncols = len(rawdat.columns)
            nrows = len(rawdat)
            self.scale[airline] = np.ones(ncols)
            rawdat = np.array(rawdat, dtype=float)
            self._normalise(airline, rawdat)
            self._split(train, test, airline, nrows)


    def _normalise(self, airline, rawdat: np.ndarray[float]):
        # normalized by the maximum value of entire matrix.
        if (self.normalize == 0):
            self.dat[airline] = rawdat
            
        if (self.normalize == 1):
            self.dat[airline] = rawdat / np.max(rawdat)
            
        # normlized by the maximum value of each row(sensor).
        if (self.normalize == 2):
            for i in range(self.cols):
                self.scale[i] = np.max(np.abs(rawdat[:,i]))
                if self.scale[i] == 0:
                    self.dat[airline][:,i] = rawdat[:,i]
                else:
                    self.dat[airline][:,i] = rawdat[:,i] / np.max(np.abs(rawdat[:,i]))


    def _split(self, train: float, test: float, airline: str, nrows: int, ncols: int):
        train: int = int(train * nrows)
        test: int = int(test * nrows)
        train_set = range(self.window + self.horizon-1, train)
        test_set = range(train, test)
        valid_set = range(test, nrows)
        self.train[airline] = self._batchify(train_set, ncols)
        self.test[airline] = self._batchify(test_set, ncols)
        self.valid[airline] = self._batchify(valid_set, ncols)
        
        
    def _batchify(self, idx_set: list[int], ncols: int) -> list[torch.Tensor]:
        n = len(idx_set)
        X = torch.zeros((n, self.window, ncols))
        Y = torch.zeros((n, ncols))
        
        #? How is the Y value selected?
        for i in range(n):
            end = idx_set[i] - self.horizon + 1
            start = end - self.window
            X[i,:,:] = torch.from_numpy(self.dat[start:end, :])
            Y[i,:] = torch.from_numpy(self.dat[idx_set[i], :]) #Y ends self.horizon steps ahead of X --> self.horizon is the forcasting horizon

        return [X, Y]


    def get_batches(self, inputs: torch.Tensor, targets: torch.Tensor, batch_size: int, shuffle: bool = True):
        '''
        Generates a batch of samples. The yield command indicates this function is used as a generator to
        iterate over a sequence of batches. 

        The function is used in the train.py file to generate a batch of samples for training the model.
        '''
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt] 
            Y = targets[excerpt]
            if (self.cuda):
                X = X.cuda()
                Y = Y.cuda()  
            yield Variable(X), Variable(Y)
            start_idx += batch_size
