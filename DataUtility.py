import pandas as pd
import numpy as np
from torch.autograd import Variable
import torch
from typing import List

def normal_std(x):
    """
    Standard Deviation with Bessels Correction, i.e. calculating the standard deviation from a sample of a population.
    """
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class DataUtility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, args, train: float, rawdat: pd.DataFrame):
        self.rawdat: np.ndarray[float] = np.array(rawdat, dtype=float)
        self.cuda: bool = args.cuda
        # self.cuda = False #! for local testing
        self.horizon: int = args.horizon
        self.window: int = args.window
        self.normalize: int = args.normalize
        self.data: np.ndarray = np.ndarray((0, 0))
        self.datetimes: list[pd.Timestamp] = []
        self.rows: int = 0
        self.cols: int = 0

        # TODO: change rawdat input to come from outside of the class
        # if args.form41:
            # self.rawdat: np.ndarray[float] = self._form41_dataloader(args.data)
        # else:
            # self.rawdat: np.ndarray[float] = self._dataloader(args.data)

        self.rows, self.cols = self.rawdat.shape
        self.dat: np.ndarray[float] = np.zeros((self.rows, self.cols))
        self.scale: np.ndarray[float] = np.ones(self.cols)
        self._normalized()
        self._split(int(train * self.rows))

        self.scale: torch.Tensor = torch.from_numpy(self.scale).float()
        tmp = self.test[1][:,-6] * self.scale.expand(self.test[1].size(0), self.cols)[:,-6] #* Uses only the profit column 
        # tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.cols) #* Uses all columns of the feature vector
        
            
        if self.cuda:
            self.scale = self.scale.cuda()
        self.scale = Variable(self.scale)
        
        self.std_data = normal_std(tmp) #Bessels corrected standard deviation of the test set (target value)
        self.mean_naive_error_data = torch.mean(torch.abs(tmp - torch.mean(tmp))) #Mean Absolute Error of the test set. It consideres the mean absolute error from the test set by usind a naive predictor (mean of the test set)


    def _normalized(self):
        if (self.normalize == 0):
            self.dat = self.rawdat
            
        # normalized by the maximum value of entire matrix.
        if (self.normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)
            
        # normlized by the maximum value of each col(sensor).
        if (self.normalize == 2):
            for i in range(self.cols):
                self.scale[i] = np.max(np.abs(self.rawdat[:,i]))
                if self.scale[i] == 0:
                    self.dat[:,i] = self.rawdat[:,i]
                else:
                    self.dat[:,i] = self.rawdat[:,i] / np.max(np.abs(self.rawdat[:,i]))
            
        
    def _split(self, train: float):
        train_set = range(self.window+self.horizon-1, train)
        test_set = range(train, self.rows)
        self.train = self._batchify(train_set)
        self.test = self._batchify(test_set)
        
        
    def _batchify(self, idx_set: int) -> List[torch.Tensor]:
        '''
        Generates the feature and target vectors for the model by considering the length of the training and test set. The feature vector is a tensor of shape (n, window, cols) and the target vector is a tensor of shape (n, cols).
        The outcome of _batchify() is used in get_batches to generate a batch of samples from each set (train and test).
        '''
        n: int = len(idx_set)
        X: torch.Tensor = torch.zeros((n,self.window,self.cols))
        Y: torch.Tensor = torch.zeros((n,self.cols))
        
        for i in range(n):
            end: int = idx_set[i] - self.horizon + 1
            start: int = end - self.window
            X[i,:,:] = torch.from_numpy(self.dat[start:end, :]) #self.dat is normalized from self.rawdat and self.rawdat is the original data given as input from the dataloader
            Y[i,:] = torch.from_numpy(self.dat[idx_set[i], :]) #Y ends self.horizon steps ahead of X --> self.horizon is the forcasting horizon

        return [X, Y]

    def get_batches(self, inputs: torch.Tensor, targets: torch.Tensor, batch_size: int, shuffle: bool = True): #? Why do we shuffle the data? Isn't the order important?
        '''
        Generates a batch of samples Each sample has the shape of (window,n_features) for X and (1,n_features) for Y. The yield command indicates this function is used as a generator to
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