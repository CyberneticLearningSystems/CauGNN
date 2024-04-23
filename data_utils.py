import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable


def normal_std(x):
    """
    Standard Deviation with Bessels Correction, i.e. calculating the standard deviation from a sample of a population.
    """
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class DataUtility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, args, train: float, test: float):
        self.cuda: bool = args.cuda
        self.horizon: int = args.horizon
        self.window: int = args.window
        self.normalize: int = args.normalize
        self.data: np.ndarray = np.ndarray((0, 0))
        self.datetimes: list[pd.Timestamp] = []
        self.rows: int = 0
        self.cols: int = 0

        if args.form41:
            self.rawdat: np.ndarray[float] = self._form41_dataloader(args.data)
        else:
            self.rawdat: np.ndarray[float] = self._dataloader(args.data)

        self.rows, self.cols = self.rawdat.shape
        self.dat: np.ndarray = np.zeros((self.rows, self.cols))
        self.scale: np.ndarray = np.ones(self.cols)
        self._normalized(self.normalize)
        self._split(int(train * self.rows), int((train+test) * self.rows))

        self.scale: torch.Tensor = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.cols)
            
        if self.cuda:
            self.scale = self.scale.cuda()
        self.scale = Variable(self.scale)
        
        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))


    def _dataloader(self, path: str):
        try:
            data = pd.read_csv(path, delimiter=',')
            self.datetimes = list(pd.to_datetime(data.pop('date')))
        except KeyError:
            data = pd.read_csv(path, delimiter=';')
            self.datetimes = list(pd.to_datetime(data.pop('date'), format='%d.%m.%Y %H:%M'))
        data = np.array(data, dtype=float)
        return data
    

    def _form41_dataloader(self, path: str):
        try:
            data = pd.read_csv(path, delimiter=',')
            self.year = data.pop('YEAR')
            self.carrier = data.pop('UNIQUE_CARRIER_NAME')
            # self.month = data.pop('MONTH')
        except KeyError:
            data = pd.read_csv(path, delimiter=';')
            self.year = data.pop('YEAR')
            self.carrier = data.pop('UNIQUE_CARRIER_NAME')
            # self.month = data.pop('MONTH')

        data.dropna(inplace=True)
        self.carrier = data.pop('AIRLINE_ID')
        data = np.array(data, dtype=float)
        return data


    def _normalized(self):
        # normalized by the maximum value of entire matrix.
        if (self.normalize == 0):
            self.dat = self.rawdat
            
        if (self.normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)
            
        # normlized by the maximum value of each row(sensor).
        if (self.normalize == 2):
            for i in range(self.cols):
                self.scale[i] = np.max(np.abs(self.rawdat[:,i]))
                if self.scale[i] == 0:
                    self.dat[:,i] = self.rawdat[:,i]
                else:
                    self.dat[:,i] = self.rawdat[:,i] / np.max(np.abs(self.rawdat[:,i]))
            
        
    def _split(self, train: float, test: float):
        train_set = range(self.window+self.horizon-1, train)
        test_set = range(train, test)
        valid_set = range(test, self.rows)
        self.train = self._batchify(train_set)
        self.test = self._batchify(test_set)
        self.valid = self._batchify(valid_set)
        
        
    def _batchify(self, idx_set: int):
        n = len(idx_set)
        X = torch.zeros((n,self.window,self.cols))
        Y = torch.zeros((n,self.cols))
        
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
