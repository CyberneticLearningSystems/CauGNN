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
        self.cuda: bool = args.cuda
        self.horizon: int = args.horizon
        self.window: int = args.window
        self.normalize: int = args.normalize
        self.args = args # TODO: remove this and check its usage
        self.Airlines: Dict[str, DataUtility] = {}
        self.timescales: Dict[str, np.ndarray] = {} #! adding timescales per airline (for plotting)

        self._airlines_with_short_operationperiods(rawdat)
        self._normalized(rawdat)
        self._airline_batching(train)
        self._airline_splitting(train)
        self._batchify_combined_airlines()

        self.scale: torch.Tensor = torch.from_numpy(self.scale).float()


    def _airlines_with_short_operationperiods(self, rawdat: pd.DataFrame):
        airlines: list[str] = rawdat['AIRLINE_ID'].unique()
        self.airlines = []
        for airline in airlines:
            airlinedat = rawdat[rawdat['AIRLINE_ID'] == airline]
            if len(airlinedat) >= 80:
                self.airlines.append(airline)

    
    def _normalized(self, rawdat: pd.DataFrame):
        data_to_scale: pd.DataFrame = rawdat[rawdat['AIRLINE_ID'].isin(self.airlines)].copy()
        self.airline_ids = data_to_scale['AIRLINE_ID'].to_numpy()
        self.timescale: str = data_to_scale['YEAR'].astype(str) + '-' + data_to_scale['QUARTER'].astype(str)
        data_to_scale: pd.DataFrame = data_to_scale.drop(columns=['YEAR', 'QUARTER', 'AIRLINE_ID'])

        self.cols: int = data_to_scale.shape[1]
        self.scale: np.ndarray[float] = np.ones(self.cols)

        # no normalisation
        if (self.normalize == 0):
            self.data_scaled = data_to_scale
            
        # normalized by the maximum value of entire matrix.
        elif (self.normalize == 1):
            self.data_scaled = data_to_scale / np.max(data_to_scale)
            
        # normlized by the maximum value of each col(sensor) PER AIRLINE.
        elif (self.normalize == 2):
            for airline in self.airlines:
                idxs = np.where(self.airline_ids == airline)
                self.data_scaled.iloc[idxs] = data_to_scale.iloc[idxs]
                for i in range(self.cols):
                    self.scale[i] = np.max(np.abs(data_to_scale.iloc[idxs][i]))
                    if self.scale[i] == 0:
                        self.data_scaled.iloc[idxs][i] = data_to_scale.iloc[idxs][i]
                    else:
                        self.data_scaled.iloc[idxs][i] = data_to_scale.iloc[idxs][i] / np.max(np.abs(data_to_scale.iloc[idxs][i]))

        # normalized by the maximum value of each col(sensor) for all airlines.
        elif (self.args.normalize == 3): 
            self.data_scaled = data_to_scale.copy()
            for i in range(self.cols):
                self.scale[i] = np.max(np.abs(data_to_scale.iloc[:,i]))
                if self.scale[i] == 0:
                    self.data_scaled.iloc[:,i] = self.data_scaled.iloc[:,i]
                else:
                    self.data_scaled.iloc[:,i] = self.data_scaled.iloc[:,i] / self.scale[i]

        # normalizes only the profit column as the rest is normalized through the pca in advance
        elif (self.args.normalize == 4): 
            self.data_scaled = data_to_scale.copy()
            self.scale[-1] = np.max(np.abs(data_to_scale))
            if self.scale[-1] == 0:
                self.data_scaled.iloc[:,-1] = self.data_scaled.iloc[:,-1]
            else:
                self.data_scaled.iloc[:,-1] = self.data_scaled.iloc[:,-1] / self.scale[-1]
            
        else:
            raise NotImplementedError('Normalisation must be set to 0, 1, 2, 3 or 4')
        
        self.data_scaled['AIRLINE_ID'] = self.airline_ids
        self.data_scaled['TIMESCALE'] = self.timescale


    def _airline_batching(self, train_size: float):
        # self.nairlines: int = 0
        # length of training / testing sets to set tensor sizes
        trainingsamples: int = 0
        testingsamples: int = 0

        # train / test ranges for each airline
        train_sets: Dict[str, range] = {}
        test_sets: Dict[str, range] = {}

        for airline in self.airlines:
            airlinedat = self.data_scaled[self.data_scaled['AIRLINE_ID'] == airline]
            n: int = len(airlinedat)
            train: int = int(n * train_size)
            
            airlinedat = airlinedat.drop(columns=['AIRLINE_ID'])
            self.timescales[airline] = airlinedat['TIMESCALE'].to_numpy()
            airlinedat = airlinedat.drop(columns=['TIMESCALE'])
            self.Airlines[airline] = DataUtility(self.args, train, airlinedat)
            # self.nairlines += 1

    
    def _airline_splitting(self, train_size: float):
        # TODO: Implement splitting of data for each airline
        train_length = 0
        train_set: Dict[str, list[int]] = {}
        test_length = 0
        test_set: Dict[str, list[int]] = {}
        timescale_train = {}
        timescale_test = {}
        for airline in self.airlines:
            n = len(self.data_scaled[self.data_scaled['AIRLINE_ID'] == airline])
            train = int(n * train_size)
            train_length += train
            train_set[airline] = range(self.window + self.horizon - 1, train_length)
            timescale_train[airline] = self.data_scaled[self.data_scaled['AIRLINE_ID'] == airline]['TIMESCALE']

            test = n - train
            test_set[airline] = range(train_length, n)
            timescale_test[airline] = self.data_scaled[self.data_scaled['AIRLINE_ID'] == airline]['TIMESCALE']
            test_length += test
        
        self.data_scaled = self.data_scaled.drop(columns=['AIRLINE_ID', 'TIMESCALE'])
        
        self.X_train: torch.Tensor = torch.zeros((train_length, self.window, self.cols-1)) # -1 to account for the timescale column
        self.Y_train: torch.Tensor = torch.zeros((train_length, self.cols))
        self.train_timescale = []
        self.train_airlineIDs = []

        self.X_test: torch.Tensor = torch.zeros((test_length, self.window, self.cols-1)) # -1 to account for the timescale column
        self.Y_test: torch.Tensor = torch.zeros((test_length, self.cols))
        self.test_timescale = []
        self.test_airlineIDs = []

        train = 0 
        test = 0
        for airline in self.airlines:
            for i in train_set[airline]:
                end: int = i - self.horizon + 1
                start: int = end - self.window
                self.X_train[train,:,:] = torch.from_numpy(self.data_scaled.iloc[start:end,:].to_numpy())
                self.Y_train[train,:] = torch.from_numpy(self.data_scaled.iloc[i,:].to_numpy())
                self.train_timescale.append(timescale_train[airline].iloc[i])
                self.train_airlineIDs.append(self.airline_ids[i])
                train += 1
            del i
            
            for i in test_set[airline]:
                end: int = i - self.horizon + 1
                start: int = end - self.window
                self.X_test[test,:,:] = torch.from_numpy(self.data_scaled.iloc[start:end,:].to_numpy())
                self.Y_test[test,:] = torch.from_numpy(self.data_scaled.iloc[i,:].to_numpy())
                self.test_timescale.append(timescale_test[airline].iloc[i])
                self.test_airlineIDs.append(self.airline_ids[i])
                test += 1
            del i

        self.train = (self.X_train, self.Y_train)
        self.test = (self.X_test, self.Y_test)


    def _batchify_combined_airlines(self):

        #Length of all training and testing samples
        length_training = 0
        length_testing = 0
        for airline in self.airlines:
            length_training += self.Airlines[airline].train[0].shape[0]
            length_testing += self.Airlines[airline].test[0].shape[0]

        #Create a tensor for all training and testing samples 
        X_train: torch.Tensor = torch.zeros((length_training, self.Airlines[self.airlines[0]].train[0].shape[1], self.Airlines[self.airlines[0]].train[0].shape[2])) #Dim: (length of all training samples x window size x number of features)
        Y_train: torch.Tensor = torch.zeros((length_training, self.Airlines[self.airlines[0]].train[1].shape[1])) #Dim: (length of all training samples x number of features)
        X_test: torch.Tensor = torch.zeros((length_testing, self.Airlines[self.airlines[0]].test[0].shape[1], self.Airlines[self.airlines[0]].test[0].shape[2])) #Dim: (length of all testing samples x window size x number of features)
        Y_test: torch.Tensor = torch.zeros((length_testing, self.Airlines[self.airlines[0]].test[1].shape[1])) #Dim: (length of all testing samples x number of features)

        j = 0
        k = 0
        for airline in self.airlines:
            for i in range(self.Airlines[airline].train[0].shape[0]):
                X_train[j,:,:] = self.Airlines[airline].train[0][i,:,:]
                Y_train[j,:] = self.Airlines[airline].train[1][i,:]
                j += 1
            del i

            for i in range(self.Airlines[airline].test[0].shape[0]):
                X_test[k,:,:] = self.Airlines[airline].test[0][i,:,:]
                Y_test[k,:] = self.Airlines[airline].test[1][i,:]
                k += 1

        self.train = (X_train, Y_train)
        self.test = (X_test, Y_test)


    def airlinedata_get_batches(self, inputs: torch.Tensor, targets: torch.Tensor, batch_size: int, shuffle: bool = True):
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
            if (self.args.cuda):
                X = X.cuda()
                Y = Y.cuda()  
            yield Variable(X), Variable(Y)
            start_idx += batch_size

    

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
        
