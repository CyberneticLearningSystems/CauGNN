import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable


def normal_std(x):
    """
    Standard Deviation with Bessels Correction, i.e. calculating the standard deviation from a sample of a population.
    """
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name: str, train: float, valid: float, cuda: bool, horizon: int, window: int, normalize: int = 2, form41: bool = False):
        self.cuda: bool = cuda
        self.window: int = window
        self.horizon: int = horizon # The number of steps ahead to predict
        self.data: np.ndarray = np.ndarray((0, 0))
        self.datetimes: list[pd.Timestamp] = []
        self.normalize: int = 2
        self.rows: int = 0
        self.cols: int = 0

        if form41:
            self.rawdat: np.ndarray[float] = self._form41_dataloader(file_name)
        else:
            self.rawdat: np.ndarray[float] = self._dataloader(file_name)

        self.dat: np.ndarray = np.zeros(self.rawdat.shape)
        self.rows, self.cols = self.dat.shape
        self.scale: np.ndarray = np.ones(self.cols)
        self._normalized(normalize)
        #! should throw an error bcs valid parameter is not given with no default
        self._split(int(train * self.rows), int((train+valid) * self.rows))
        # fin = open(file_name)
        # self.rawdat = np.loadtxt(fin, delimiter=',', usecols=list(range(1, len(fin.readlines))))
        # self.dat = np.zeros(self.rawdat.shape)
        # self.rows, self.cols = self.dat.shape

        self.scale = torch.from_numpy(self.scale).float()
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
            self.carrier = data.pop('AIRLINE_ID')
            self.year = data.pop('YEAR')
            self.month = data.pop('MONTH')
        except KeyError:
            data = pd.read_csv(path, delimiter=';')
            self.carrier = data.pop('AIRLINE_ID')
            self.year = data.pop('YEAR')
            self.month = data.pop('MONTH')
        data = np.array(data, dtype=float)
        return data
    

    def _normalized(self, normalize: int):
        # normalized by the maximum value of entire matrix.
        if (normalize == 0):
            self.dat = self.rawdat
            
        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)
            
        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.cols):
                self.scale[i] = np.max(np.abs(self.rawdat[:,i]))
                self.dat[:,i] = self.rawdat[:,i] / np.max(np.abs(self.rawdat[:,i]))
            
        
    def _split(self, train: float, valid: float):
        train_set = range(self.window+self.horizon-1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.rows)
        self.train = self._batchify(train_set)
        self.valid = self._batchify(valid_set)
        self.test = self._batchify(test_set)
        
        
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
            # print('Y',self.dat[idx_set[i], :])

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


class STS_Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_train, file_test, cuda):
        self.cuda = cuda
        self.x_train,self.y_train = self.loaddata(file_train)
        self.x_test,self.y_test = self.loaddata((file_test))
        self.num_nodes = len(self.x_train.shape[0])+len(self.x_test.shape[0])
        self.num_class = len(np.unique(self.y_test))
        self.batch_size = min(self.x_train.shape[0] / 10, 16)

        x_train_mean = self.x_train.mean()
        x_train_std = self.x_train.std()
        self.x_train = (self.x_train - x_train_mean) / (x_train_std)
        self.x_test = (self.x_test - x_train_mean) / (x_train_std)

        self.y_train = (self.y_train - self.y_train.min()) / (self.y_train.max() - self.y_train.min()) * (self.num_class - 1)
        self.y_test = (self.y_test - self.y_test.min()) / (self.y_test.max() - self.y_test.min()) * (self.num_class - 1)

    def loaddata(self,filename):
        data = np.loadtxt(filename, delimiter=',')
        Y = data[:, 0]
        X = data[:, 1:]
        return X, Y

    def get_batches(self, inputs, targets, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        end_idx = length
        excerpt = index[start_idx:end_idx]
        X = inputs[excerpt]
        Y = targets[excerpt]
        if (self.cuda):
            X = X.cuda()
            Y = Y.cuda()
        return Variable(X), Variable(Y)
